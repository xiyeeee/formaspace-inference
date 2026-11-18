import asyncio
from contextlib import asynccontextmanager
import hmac
import io
import json
import logging
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .parameters import TaskParameters
from .store import ServiceError, TERMINAL, task_path
from .subscriptions import SlowSubscriber
from .worker import TaskManager


LOGGER = logging.getLogger(__name__)


class UploadTooLarge(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="Upload exceeds service limit")


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise UploadTooLarge()
            return message

        # Starlette's multipart parsing maps this into the registered exception handler.
        await self.app(scope, bounded_receive, send)


def authenticated(header, token):
    parts = (header or "").split()
    return (len(parts) == 2 and parts[0].lower() == "bearer"
            and hmac.compare_digest(parts[1].encode(), token.encode()))


def validate_image(data, settings):
    if len(data) > settings.max_upload_bytes:
        raise ServiceError(413, "IMAGE_TOO_LARGE", "图片超过上传限制")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG") or image.width * image.height > settings.max_image_pixels:
                raise ValueError()
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except Exception:
        raise ServiceError(422, "INVALID_IMAGE", "图片无法解码、格式不支持或像素超限") from None


def create_app(settings=None, engine_factory=None):
    settings = settings or Settings.from_env()
    manager = TaskManager(settings, engine_factory)

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(manager.start)
        try:
            yield
        finally:
            await run_in_threadpool(manager.stop)

    app = FastAPI(title="FormaSpace inference v1", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + 65536)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"error": {"code": "INVALID_REQUEST", "message": "请求字段无效"}}, status_code=422)

    @app.exception_handler(UploadTooLarge)
    async def upload_too_large(request, error):
        return JSONResponse({"error": {"code": "IMAGE_TOO_LARGE", "message": "上传内容超过限制"}}, status_code=413)

    @app.middleware("http")
    async def authorize(request: Request, call_next):
        if request.url.path not in ("/healthz", "/readyz"):
            if not authenticated(request.headers.get("authorization"), settings.token):
                return JSONResponse({"error": {"code": "UNAUTHORIZED", "message": "服务令牌无效"}}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        return {"alive": True}

    @app.get("/readyz")
    async def readyz():
        status = manager.readiness()
        return JSONResponse(status, status_code=200 if status["ready"] else 503)

    @app.post("/api/v1/tasks", status_code=202)
    def submit(taskId: UUID = Form(...), image: UploadFile = File(...), parameters: str = Form("{}")):
        # Sync endpoint runs in Starlette's threadpool: PIL, upload reads and fsync never block asyncio.
        if len(parameters) > 4096:
            raise ServiceError(422, "INVALID_PARAMETERS", "参数内容过长")
        try:
            parsed = TaskParameters.model_validate_json(parameters).model_dump(exclude_none=True)
        except (ValidationError, ValueError):
            raise ServiceError(422, "INVALID_PARAMETERS", "推理参数不支持或超出范围") from None
        data = image.file.read(settings.max_upload_bytes + 1)
        validate_image(data, settings)
        task, created = manager.submit(str(taskId), parsed, data)
        snapshot = manager.store.snapshot(task.task_id)
        return {"taskId": task.task_id, "status": "queued" if created else snapshot["status"],
                "snapshotPath": task_path(task.task_id), "eventsPath": task_path(task.task_id) + "/events"}

    @app.get("/api/v1/tasks/{task_id}")
    def snapshot(task_id: UUID):
        return manager.store.snapshot(str(task_id))

    @app.get("/api/v1/tasks/{task_id}/previews/{version}")
    def preview(task_id: UUID, version: int):
        state = manager.store.snapshot(str(task_id))
        latest = manager.store.preview_metadata(str(task_id), version)
        if not latest:
            raise ServiceError(404, "PREVIEW_NOT_FOUND", "点云尚未就绪或版本不存在")
        path = manager.store.directory(str(task_id)) / f"preview_step_{latest['sourceStep']:03d}.json"
        if not path.is_file():
            raise ServiceError(404, "PREVIEW_NOT_FOUND", "点云文件不存在")
        data = json.loads(path.read_text())
        return {"taskId": str(task_id), "previewVersion": version, "sourceStep": latest["sourceStep"],
                "pointCount": latest["pointCount"], "positions": data["positions"]}

    @app.get("/api/v1/tasks/{task_id}/result")
    def result(task_id: UUID):
        state = manager.store.snapshot(str(task_id))
        path = manager.store.directory(str(task_id)) / "final.glb"
        if state["status"] != "completed" or not path.is_file():
            raise ServiceError(409, "RESULT_NOT_READY", "模型结果尚未就绪")
        return FileResponse(path, media_type="model/gltf-binary", filename="final.glb")

    @app.websocket("/api/v1/tasks/{task_id}/events")
    async def events(websocket: WebSocket, task_id: str):
        if not authenticated(websocket.headers.get("authorization"), settings.token):
            await websocket.close(code=4401)
            return
        try:
            task_id = str(UUID(task_id))
            control, subscription = manager.store.subscribe(task_id, asyncio.get_running_loop())
        except (ValueError, ServiceError):
            await websocket.close(code=4404)
            return
        workers = []
        try:
            await websocket.accept()
            await asyncio.wait_for(websocket.send_json(control), timeout=settings.send_timeout)
            if control["payload"]["status"] in TERMINAL:
                await websocket.close(code=1000)
                return

            async def send_events():
                while True:
                    event = await subscription.next()
                    await asyncio.wait_for(websocket.send_json(event), timeout=settings.send_timeout)
                    if event["type"] in ("task_completed", "task_failed"):
                        await websocket.close(code=1000)
                        return

            async def receive_disconnect():
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return

            workers = [asyncio.create_task(send_events()), asyncio.create_task(receive_disconnect())]
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                future.result()
        except (SlowSubscriber, asyncio.TimeoutError):
            await websocket.close(code=1013, reason="Subscriber overflow or send timeout; reconnect for snapshot")
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            manager.store.unsubscribe(task_id, subscription)
            for future in workers:
                future.cancel()
            try:
                await asyncio.gather(*workers, return_exceptions=True)
            except asyncio.CancelledError:
                pass

    return app
