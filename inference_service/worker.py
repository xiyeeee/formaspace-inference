import logging
import queue
import threading

from .engine import HunyuanEngine
from .store import ServiceError, TaskStore, TERMINAL, request_fingerprint


LOGGER = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, settings, engine_factory=None):
        self.settings = settings
        self.store = TaskStore(settings)
        self.queue = queue.Queue(maxsize=settings.queue_capacity)
        self.engine_factory = engine_factory or (lambda: HunyuanEngine(settings.device))
        self.stopping = threading.Event()
        self.loaded = threading.Event()
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._work, name="inference-worker", daemon=True)

    def start(self):
        try:
            self.store.initialize()
            self.thread.start()
        except Exception:
            self.store.close()
            raise

    def readiness(self):
        return {"ready": self.loaded.is_set() and self.thread.is_alive() and not self.failed.is_set()
                and not self.stopping.is_set() and not self.queue.full(),
                "modelId": "tencent/Hunyuan3D-2mini", "device": self.settings.device,
                "queueDepth": self.queue.qsize()}

    def submit(self, task_id, parameters, image_bytes):
        # Validation already ran off-loop. Claim, durable create and enqueue are atomic for duplicate POSTs.
        with self.store.lock:
            existing = self.store.tasks.get(task_id)
            if existing is not None:
                fingerprint = request_fingerprint(parameters, image_bytes)
                if existing.fingerprint != fingerprint:
                    raise ServiceError(409, "IDEMPOTENCY_CONFLICT", "taskId 已用于不同的图片或参数")
                return existing, False
            if not self.loaded.is_set() or self.failed.is_set() or self.stopping.is_set() or not self.thread.is_alive():
                raise ServiceError(503, "INFERENCE_UNAVAILABLE", "模型尚未就绪或推理 worker 不可用")
            if self.queue.full():
                raise ServiceError(429, "QUEUE_FULL", "推理队列已满，请稍后使用同一 taskId 重试")
            task, _ = self.store.create(task_id, parameters, image_bytes)
            self.queue.put_nowait(task_id)
            return task, True

    def _work(self):
        try:
            engine = self.engine_factory()
            self.loaded.set()
            while not self.stopping.is_set():
                try:
                    task_id = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    task = self.store.find(task_id)
                    self.store.emit(task_id, "task_started", "preprocessing")
                    engine.run(task, self.store)
                    if self.store.snapshot(task_id)["status"] not in TERMINAL:
                        raise RuntimeError("Engine returned without a terminal event")
                except Exception as error:
                    from hy3dgen.shapegen.preview import is_device_failure

                    LOGGER.exception("Inference failed: taskId=%s", task_id)
                    self.store.emit(task_id, "task_failed", "failed", payload={
                        "code": "INFERENCE_FAILED", "message": "模型推理失败，请检查推理服务日志",
                    })
                    if is_device_failure(error):
                        raise
                finally:
                    self.queue.task_done()
        except BaseException:
            self.failed.set()
            LOGGER.exception("Inference worker unavailable; restart service before accepting new work")
            with self.store.lock:
                waiting = [task.task_id for task in self.store.tasks.values() if task.snapshot["status"] not in TERMINAL]
            for task_id in waiting:
                self.store.emit(task_id, "task_failed", "failed", payload={
                    "code": "WORKER_UNAVAILABLE", "message": "推理 worker 不可用，请重启服务",
                })
        finally:
            self.loaded.clear()

    def stop(self):
        self.stopping.set()
        # Graceful shutdown finishes the active task. Queued work is failed on next startup.
        self.thread.join()
        self.store.close()
