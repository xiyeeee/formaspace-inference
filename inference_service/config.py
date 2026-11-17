from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    token: str
    root: Path
    device: str = "mps"
    queue_capacity: int = 4
    subscriber_capacity: int = 64
    subscriber_limit: int = 32
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    send_timeout: float = 10.0

    def __post_init__(self):
        if not self.token.strip():
            raise ValueError("INFERENCE_API_TOKEN must be configured")
        if min(self.queue_capacity, self.subscriber_capacity, self.subscriber_limit,
               self.max_upload_bytes, self.max_image_pixels, self.send_timeout) <= 0:
            raise ValueError("Service capacity and limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            token=os.environ.get("INFERENCE_API_TOKEN", ""),
            root=Path(os.environ.get("INFERENCE_STORAGE_ROOT", "~/.formaspace/inference-tasks")).expanduser().resolve(),
            device=os.environ.get("INFERENCE_DEVICE", "mps"),
            queue_capacity=int(os.environ.get("INFERENCE_QUEUE_CAPACITY", "4")),
            subscriber_capacity=int(os.environ.get("INFERENCE_SUBSCRIBER_CAPACITY", "64")),
            subscriber_limit=int(os.environ.get("INFERENCE_SUBSCRIBER_LIMIT", "32")),
        )
