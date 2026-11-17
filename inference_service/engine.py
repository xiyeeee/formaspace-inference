import json

from .store import task_path


class ServiceJournal:
    """Adapt shared probe events to the public v1 wire protocol; never expose paths/errors."""
    def __init__(self, store, task):
        self.store = store
        self.task_id = task.task_id
        self.completed_steps = 0
        self.preview_version = 0

    def emit(self, event_type, stage, **extra):
        if event_type == "task_started":
            return  # Worker already persisted task_started/preprocessing before image preparation.
        payload = {}
        if event_type == "preview_saved":
            event_type, stage = "preview_ready", "sampling"
            points = json.loads((self.store.directory(self.task_id)
                                 / f"preview_step_{self.completed_steps:03d}.json").read_text())["positions"]
            payload = {
                "sourceStep": self.completed_steps, "pointCount": len(points), "positions": points,
                "positionsPath": task_path(self.task_id) + f"/previews/{self.preview_version}",
                "decodeSeconds": extra["decodeSeconds"],
            }
        elif event_type == "preview_failed":
            stage = "sampling"
            payload = {"code": "PREVIEW_DECODE_FAILED", "message": "本次中间预览生成失败",
                       "fatal": extra.get("fatal", False)}
        elif event_type == "task_completed":
            payload = {"modelPath": task_path(self.task_id) + "/result", "format": "glb"}
        elif event_type == "task_failed":
            payload = {"code": "INFERENCE_FAILED", "message": "模型推理失败，请检查推理服务日志"}
        elif event_type == "sampling_step":
            payload = {"timestep": extra["timestep"]}
        self.store.emit(self.task_id, event_type, stage, completed_steps=self.completed_steps,
                        preview_version=self.preview_version, payload=payload)

    def close(self):
        pass  # Store owns the persistent journal; subscribers may outlive the worker invocation.


class HunyuanEngine:
    def __init__(self, device):
        # Called exactly once, inside the sole inference worker thread.
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self.device = device
        self.remover = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini",
            device=device, use_safetensors=True,
        )

    def run(self, task, store):
        from hy3dgen.shapegen.preview import PreviewProbe, ProbeConfig, prepare_image

        parameters = task.parameters
        config = ProbeConfig(
            seed=parameters["seed"], steps=parameters["steps"], preview_step=parameters["previewStep"],
            device=self.device, guidance_scale=parameters["guidanceScale"],
            final_resolution=parameters["finalResolution"], preview_resolution=parameters["previewResolution"],
            num_chunks=parameters["numChunks"], point_count=parameters["pointCount"],
        )
        directory = store.directory(task.task_id)
        image = prepare_image(directory / "input.image", self.remover)
        PreviewProbe(config, directory, enabled=parameters["previewEnabled"],
                     journal=ServiceJournal(store, task)).run(self.pipeline, image)
