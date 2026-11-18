"""Shared synchronous inference/preview implementation for CLI and task service."""

from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import pickle
import random
import time
import uuid

import numpy as np
from PIL import Image
import torch
import trimesh

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 1234
    steps: int = 30
    preview_step: int = 24
    model_id: str = "tencent/Hunyuan3D-2mini"
    subfolder: str = "hunyuan3d-dit-v2-mini"
    device: str = "mps"
    guidance_scale: float = 5.0
    final_resolution: int = 256
    preview_resolution: int = 128
    num_chunks: int = 4000
    point_count: int = 5000


class UnsafePreviewError(RuntimeError):
    """Device failure or state change: do not continue sampling this pipeline."""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EventJournal:
    def __init__(self, directory, total_steps):
        directory.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite a prior task: this probe does not implement resume.
        self.stream = (directory / "events.jsonl").open("x", encoding="utf-8")
        self.task_id = str(uuid.uuid4())
        self.sequence = 0
        self.preview_version = 0
        self.completed_steps = 0
        self.total_steps = total_steps

    def emit(self, event_type, stage, **extra):
        self.sequence += 1
        event = {
            "taskId": self.task_id, "sequence": self.sequence, "type": event_type,
            "stage": stage, "completedSteps": self.completed_steps,
            "totalSteps": self.total_steps, "previewVersion": self.preview_version,
            **extra,
        }
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def random_fingerprints(device):
    """Observe, never seed or restore any global generator."""
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torchCpu": torch.get_rng_state().numpy().tobytes()}
    if str(device).startswith("mps"):
        states["torchDevice"] = torch.mps.get_rng_state().cpu().numpy().tobytes()
    elif str(device).startswith("cuda"):
        states["torchDevice"] = torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
    return {name: hashlib.sha256(pickle.dumps(state)).hexdigest() for name, state in states.items()}


def is_device_failure(error):
    message = str(error).lower()
    return isinstance(error, (UnsafePreviewError, torch.OutOfMemoryError, MemoryError)) or any(
        marker in message for marker in (
            "out of memory", "mps backend", "mps device", "metal", "command buffer",
            "device-side", "device lost", "cuda error", "cuda out of memory",
        )
    )


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def require_mesh(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1 or outputs[0] is None:
        raise ValueError("Decoder did not return one nonempty mesh")
    mesh = outputs[0]
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Mesh has no XYZ vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Mesh has no triangular faces")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains nonfinite coordinates")
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh contains invalid vertex indices")
    return mesh


def sample_surface(mesh, count, rng):
    """Area-weighted barycentric sampling in the untouched model coordinate system."""
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[mesh.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    total_area = areas.sum()
    if not np.isfinite(areas).all() or not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Mesh has no finite, positive surface area")
    selected = triangles[rng.choice(len(triangles), size=count, p=areas / total_area)]
    root = np.sqrt(rng.random((count, 1)))
    fraction = rng.random((count, 1))
    points = ((1 - root) * selected[:, 0] + root * (1 - fraction) * selected[:, 1]
              + root * fraction * selected[:, 2])
    if not np.isfinite(points).all():
        raise ValueError("Surface sampling produced nonfinite coordinates")
    return points


def save_points(directory, step, points):
    stem = f"preview_step_{step:03d}"
    ply_path = directory / f"{stem}.ply"
    json_path = directory / f"{stem}.json"
    try:
        # ASCII PLY keeps float64 coordinates; no centering, scaling or mesh processing.
        with ply_path.with_suffix(".ply.tmp").open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\nelement vertex " + str(len(points))
                         + "\nproperty double x\nproperty double y\nproperty double z\nend_header\n")
            np.savetxt(stream, points, fmt="%.17g")
        ply_path.with_suffix(".ply.tmp").replace(ply_path)
        write_json(json_path, {"positions": points.tolist()})
    except Exception:
        for path in (ply_path, json_path, ply_path.with_suffix(".ply.tmp"), json_path.with_suffix(".json.tmp")):
            path.unlink(missing_ok=True)
        raise
    return ply_path, json_path


class PreviewProbe:
    def __init__(self, config, directory, enabled=True, journal=None):
        self.config = config
        self.directory = directory
        self.enabled = enabled
        self.events = journal if journal is not None else EventJournal(directory, config.steps)
        self.preview = {"status": "pending" if enabled else "disabled", "actualStep": None,
                        "pointCount": 0, "elapsedSeconds": None, "decodeSeconds": None,
                        "randomStateUnchanged": None, "changedRandomGenerators": []}

    def decode_preview(self, pipeline, outputs):
        config = self.config
        started = time.perf_counter()
        self.preview["actualStep"] = self.events.completed_steps
        self.events.emit("preview_started", "preview_decoding")
        before = None
        diagnostics = io.StringIO()
        try:
            before = random_fingerprints(config.device)
            # Only the selected step clones the post-scheduler sample. Never reinterpret it as XYZ.
            snapshot = outputs.prev_sample.detach().clone()
            synchronize(config.device)
            decode_started = time.perf_counter()
            # _export applies 1 / vae.scale_factor, vae.forward and latents2mesh in that order.
            with redirect_stderr(diagnostics):
                decoded = pipeline._export(
                    latents=snapshot, output_type="trimesh", box_v=1.01, mc_level=0.0,
                    num_chunks=config.num_chunks, octree_resolution=config.preview_resolution,
                    mc_algo="mc", enable_pbar=False,
                )
            synchronize(config.device)
            self.preview["decodeSeconds"] = time.perf_counter() - decode_started
            # SurfaceExtractor may swallow an exception and return [None]. Preserve its traceback.
            if is_device_failure(diagnostics.getvalue()):
                raise UnsafePreviewError("Device failure reported inside surface extraction")
            mesh = require_mesh(decoded)
            points = sample_surface(mesh, config.point_count, np.random.default_rng(config.seed))
            if not torch.equal(snapshot, outputs.prev_sample):
                raise UnsafePreviewError("Preview decoding modified its latent input")
            after = random_fingerprints(config.device)
            self.preview["changedRandomGenerators"] = [name for name in before if before[name] != after[name]]
            self.preview["randomStateUnchanged"] = not self.preview["changedRandomGenerators"]
            if not self.preview["randomStateUnchanged"]:
                raise UnsafePreviewError("Global random state changed during preview: "
                                         + ", ".join(self.preview["changedRandomGenerators"]))
            ply_path, json_path = save_points(self.directory, self.events.completed_steps, points)
            self.preview.update(status="succeeded", pointCount=len(points),
                                elapsedSeconds=time.perf_counter() - started)
            self.events.preview_version += 1
            self.events.emit("preview_saved", "preview_ready", plyPath=str(ply_path.resolve()),
                             jsonPath=str(json_path.resolve()), pointCount=len(points),
                             decodeSeconds=self.preview["decodeSeconds"],
                             elapsedSeconds=self.preview["elapsedSeconds"])
        except Exception as error:
            self.preview.update(status="failed", elapsedSeconds=time.perf_counter() - started,
                                error=str(error))
            LOGGER.exception("Preview failed at completed step %s", self.events.completed_steps)
            try:
                self.preview["randomStateUnchanged"] = before is not None and before == random_fingerprints(config.device)
            except Exception:
                LOGGER.exception("Unable to inspect device RNG after preview failure")
                self.preview["randomStateUnchanged"] = False
            fatal = is_device_failure(error) or not self.preview["randomStateUnchanged"]
            self.events.emit("preview_failed", "preview_failed", error=str(error), fatal=fatal)
            if fatal:
                raise UnsafePreviewError("Preview device/state failure; sampling cannot safely continue") from error
        finally:
            if diagnostics.getvalue():
                LOGGER.error("Decoder diagnostics (including swallowed traceback):\n%s", diagnostics.getvalue())

    def run(self, pipeline, image):
        started = time.perf_counter()
        config = self.config
        summary = {"taskId": self.events.task_id, "model": asdict(config), "seed": config.seed,
                   "preview": self.preview, "finalStatus": "running",
                   "synchronousPreviewPausesSampling": True, "previewEnabled": self.enabled}
        self.events.emit("task_started", "sampling", previewEnabled=self.enabled)

        def callback(step_index, timestep, outputs):
            # Local pipeline passes i // scheduler.order after scheduler.step; supported order is 1.
            completed = int(step_index) + 1
            if completed != self.events.completed_steps + 1:
                raise RuntimeError("Unexpected callback indexing: expected consecutive zero-based indices")
            self.events.completed_steps = completed
            self.events.emit("sampling_step", "sampling", timestep=float(timestep.item()))
            LOGGER.info("%s completed %d/%d", self.events.task_id, completed, config.steps)
            if self.enabled and completed == config.preview_step:
                self.decode_preview(pipeline, outputs)
            if completed == config.steps:
                self.events.emit("finalizing", "final_decoding")

        try:
            if pipeline.scheduler.order != 1:
                raise ValueError("This probe requires scheduler.order=1 to count actual completed steps")
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            outputs = pipeline(
                image=image, num_inference_steps=config.steps, guidance_scale=config.guidance_scale,
                generator=generator, octree_resolution=config.final_resolution, num_chunks=config.num_chunks,
                mc_algo="mc", output_type="trimesh", callback=callback, callback_steps=1, enable_pbar=False,
            )
            synchronize(config.device)
            if self.events.completed_steps != config.steps:
                raise RuntimeError("Pipeline returned before all expected callbacks")
            if self.enabled and self.preview["status"] == "pending":
                raise RuntimeError("Requested preview callback was not reached")
            mesh = require_mesh(outputs)
            final_path = self.directory / "final.glb"
            temporary = self.directory / "final.glb.tmp"
            try:
                mesh.export(temporary, file_type="glb", include_normals=False)
                temporary.replace(final_path)
            finally:
                temporary.unlink(missing_ok=True)
            summary.update(finalStatus="succeeded", finalPath=str(final_path.resolve()),
                           vertexCount=len(mesh.vertices), faceCount=len(mesh.faces))
            self.events.emit("task_completed", "completed", finalPath=str(final_path.resolve()))
            return mesh
        except Exception as error:
            summary.update(finalStatus="failed", error=str(error))
            LOGGER.exception("Final generation failed")
            self.events.emit("task_failed", "failed", error=str(error))
            raise
        finally:
            summary["totalSeconds"] = time.perf_counter() - started
            write_json(self.directory / "summary.json", summary)
            self.events.close()


def prepare_image(path, remover=None):
    from hy3dgen.rembg import BackgroundRemover

    # Gradio's gr.Image(image_mode='RGBA', type='pil') precedes _gen_shape.
    with Image.open(path) as source:
        image = source.convert("RGBA")
    # Match the default check_box_rembg=True path in _gen_shape. ImageProcessorV2
    # (crop/recenter/resize/normalize) is still invoked by pipeline.prepare_image.
    return (remover if remover is not None else BackgroundRemover())(image.convert("RGB"))
