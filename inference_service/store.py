from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from uuid import UUID

from .subscriptions import Subscription


TERMINAL = {"completed", "failed"}


class ServiceError(Exception):
    def __init__(self, status, code, message):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def task_path(task_id):
    return f"/api/v1/tasks/{task_id}"


def request_fingerprint(parameters, image_bytes):
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(hashlib.sha256(image_bytes).digest() + canonical).hexdigest()


@dataclass
class Task:
    task_id: str
    fingerprint: str
    parameters: dict
    snapshot: dict
    subscribers: set = field(default_factory=set)


class TaskStore:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.root
        self.lock = threading.RLock()
        self.tasks = {}
        self.lock_file = None

    def directory(self, task_id):
        return self.root / str(UUID(task_id))

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = (self.root / ".service.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError("Another service process owns this storage root") from None
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.is_symlink() or not (directory / "request.json").is_file():
                continue
            try:
                task_id = str(UUID(directory.name))
            except ValueError:
                continue
            if task_id != directory.name:
                continue
            request = json.loads((directory / "request.json").read_text())
            task = self.new_task(task_id, request["fingerprint"], request["parameters"])
            journal = directory / "events.jsonl"
            if journal.exists():
                # A torn final line is an interrupted append, never an event to replay.
                with journal.open("rb+") as stream:
                    valid_end = 0
                    while line := stream.readline():
                        if not line.endswith(b"\n"):
                            stream.truncate(valid_end)
                            break
                        event = json.loads(line)
                        if event["taskId"] != task_id or event["sequence"] != task.snapshot["sequence"] + 1:
                            raise RuntimeError("Invalid persisted event sequence")
                        task.snapshot = self.reduce(task.snapshot, event)
                        valid_end = stream.tell()
            self.tasks[task_id] = task
            if task.snapshot["status"] not in TERMINAL:
                self.emit(task_id, "task_failed", "failed", payload={
                    "code": "SERVICE_RESTARTED", "message": "推理服务已重启，原任务已中断",
                })
            else:
                atomic_json(directory / "snapshot.json", task.snapshot)

    def new_task(self, task_id, fingerprint, parameters):
        return Task(task_id, fingerprint, parameters, {
            "taskId": task_id, "status": "queued", "sequence": 0, "stage": "queued",
            "completedSteps": 0, "totalSteps": parameters["steps"], "previewVersion": 0,
            "latestPreview": None, "result": None, "error": None,
        })

    def find(self, task_id):
        task = self.tasks.get(task_id)
        if task is None:
            raise ServiceError(404, "TASK_NOT_FOUND", "任务不存在")
        return task

    def snapshot(self, task_id):
        with self.lock:
            return deepcopy(self.find(task_id).snapshot)

    def create(self, task_id, parameters, image_bytes):
        fingerprint = request_fingerprint(parameters, image_bytes)
        existing = self.tasks.get(task_id)
        if existing:
            if existing.fingerprint != fingerprint:
                raise ServiceError(409, "IDEMPOTENCY_CONFLICT", "taskId 已用于不同的图片或参数")
            return existing, False
        temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=self.root))
        try:
            with (temporary / "input.image").open("wb") as output:
                output.write(image_bytes)
                output.flush()
                os.fsync(output.fileno())
            atomic_json(temporary / "request.json", {"fingerprint": fingerprint, "parameters": parameters})
            temporary.rename(self.directory(task_id))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        task = self.new_task(task_id, fingerprint, parameters)
        self.tasks[task_id] = task
        self.emit(task_id, "task_queued", "queued")
        return task, True

    @staticmethod
    def reduce(snapshot, event):
        updated = {**snapshot, **{key: event[key] for key in (
            "sequence", "stage", "completedSteps", "totalSteps", "previewVersion")}}
        kind = event["type"]
        payload = event["payload"]
        if kind == "task_started":
            updated["status"] = "running"
        elif kind == "preview_ready":
            updated["latestPreview"] = {
                "version": event["previewVersion"], "sourceStep": payload["sourceStep"],
                "pointCount": payload["pointCount"], "positionsPath": payload["positionsPath"],
            }
        elif kind == "task_completed":
            updated.update(status="completed", result=payload, error=None)
        elif kind == "task_failed":
            updated.update(status="failed", error=payload, result=None)
        return updated

    def emit(self, task_id, kind, stage, completed_steps=None, preview_version=None, payload=None):
        with self.lock:
            task = self.find(task_id)
            previous = task.snapshot
            if previous["status"] in TERMINAL:
                return
            event = {
                "protocolVersion": 1, "taskId": task_id, "sequence": previous["sequence"] + 1,
                "type": kind, "stage": stage,
                "completedSteps": previous["completedSteps"] if completed_steps is None else completed_steps,
                "totalSteps": previous["totalSteps"],
                "previewVersion": previous["previewVersion"] if preview_version is None else preview_version,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "payload": payload or {},
            }
            serialized = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode()
            with (self.directory(task_id) / "events.jsonl").open("ab") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            task.snapshot = self.reduce(previous, event)
            atomic_json(self.directory(task_id) / "snapshot.json", task.snapshot)
            for subscriber in tuple(task.subscribers):
                subscriber.offer(event)

    def preview_metadata(self, task_id, version):
        # Journal is authoritative, persisted before notification; preserve every successful version.
        with self.lock:
            self.find(task_id)
            if version < 1:
                return None
            with (self.directory(task_id) / "events.jsonl").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["type"] == "preview_ready" and event["previewVersion"] == version:
                        return {"version": version, **{key: event["payload"][key]
                                for key in ("sourceStep", "pointCount", "positionsPath")}}
            return None

    def subscribe(self, task_id, loop):
        # Capture snapshot and register future events in the same critical section.
        with self.lock:
            task = self.find(task_id)
            count = sum(len(item.subscribers) for item in self.tasks.values())
            if count >= self.settings.subscriber_limit:
                raise ServiceError(429, "SUBSCRIBER_LIMIT", "订阅连接已满")
            subscription = Subscription(self.settings.subscriber_capacity, loop)
            task.subscribers.add(subscription)
            snapshot = deepcopy(task.snapshot)
            control = {
                "protocolVersion": 1, "taskId": task_id, "sequence": snapshot["sequence"],
                "type": "task_snapshot", "stage": snapshot["stage"],
                "completedSteps": snapshot["completedSteps"], "totalSteps": snapshot["totalSteps"],
                "previewVersion": snapshot["previewVersion"],
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "payload": snapshot,
            }
            return control, subscription

    def unsubscribe(self, task_id, subscription):
        with self.lock:
            self.find(task_id).subscribers.discard(subscription)
            subscription.close()

    def close(self):
        if self.lock_file and not self.lock_file.closed:
            self.lock_file.close()
