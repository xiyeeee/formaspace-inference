#!/usr/bin/env python3
"""Standalone synchronous intermediate-mesh probe; no Gradio/Java/UI changes."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Re-export the original helpers for existing probe callers.
from hy3dgen.shapegen.preview import (
    EventJournal, PreviewProbe, ProbeConfig, UnsafePreviewError, prepare_image,
    random_fingerprints, require_mesh, sample_surface, save_points, write_json,
)

LOGGER = logging.getLogger("preview_probe")


def compare_meshes(baseline, preview):
    from scipy.spatial import cKDTree

    vertices_equal = (baseline.vertices.shape == preview.vertices.shape
                      and np.array_equal(baseline.vertices, preview.vertices))
    topology_equal = np.array_equal(baseline.faces, preview.faces)
    diagonal = float(np.linalg.norm(np.ptp(baseline.vertices, axis=0)))
    if diagonal <= 0:
        raise ValueError("Cannot compare a degenerate final mesh")
    distances = np.concatenate((cKDTree(baseline.vertices).query(preview.vertices)[0],
                                cKDTree(preview.vertices).query(baseline.vertices)[0])) / diagonal
    # Vertex-set distance, not an assertion of full topological/surface equivalence.
    p99 = float(np.quantile(distances, 0.99))
    return {"verticesExactlyEqual": bool(vertices_equal), "facesExactlyEqual": bool(topology_equal),
            "baselineVertexCount": len(baseline.vertices), "previewVertexCount": len(preview.vertices),
            "baselineFaceCount": len(baseline.faces), "previewFaceCount": len(preview.faces),
            "normalizedVertexDistanceMean": float(distances.mean()),
            "normalizedVertexDistanceP99": p99, "normalizedVertexDistanceMax": float(distances.max()),
            "closeByVertexMetric": p99 <= 0.005,
            "metric": "symmetric nearest-vertex distances / baseline bounding-box diagonal; p99 <= 0.005"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--preview-step", type=int, default=24)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-preview", action="store_true")
    modes.add_argument("--compare", action="store_true", help="Run baseline then preview, loading the model once")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.preview_step <= args.steps or not 0 <= args.seed < 2**63:
        parser.error("Require steps >= 1, 1 <= preview-step <= steps and 0 <= seed < 2**63")
    if not args.image.is_file():
        parser.error("Input image does not exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty; task resume is not supported")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.output_dir / "probe.log"), logging.StreamHandler(sys.stdout)])
    config = ProbeConfig(seed=args.seed, steps=args.steps, preview_step=args.preview_step, device=args.device)
    started = time.perf_counter()
    LOGGER.info("Synchronous preview pauses sampling and increases total elapsed time")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        image = prepare_image(args.image)
        load_started = time.perf_counter()
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            config.model_id, subfolder=config.subfolder, device=config.device, use_safetensors=True,
        )
        load_seconds = time.perf_counter() - load_started
        LOGGER.info("Model loaded once in %.3fs; texture disabled", load_seconds)
        if args.compare:
            baseline = PreviewProbe(config, args.output_dir / "baseline", enabled=False).run(pipeline, image)
            preview = PreviewProbe(config, args.output_dir / "with_preview").run(pipeline, image)
            comparison = compare_meshes(baseline, preview)
            write_json(args.output_dir / "comparison.json", comparison)
            write_json(args.output_dir / "summary.json", {
                "model": asdict(config), "image": str(args.image.resolve()), "seed": config.seed,
                "baseline": json.loads((args.output_dir / "baseline/summary.json").read_text()),
                "withPreview": json.loads((args.output_dir / "with_preview/summary.json").read_text()),
                "comparison": comparison, "modelLoadSeconds": load_seconds,
                "totalSeconds": time.perf_counter() - started,
                "finalStatus": "succeeded",
            })
            LOGGER.info("Comparison: %s", comparison)
        else:
            PreviewProbe(config, args.output_dir, enabled=not args.no_preview).run(pipeline, image)
            summary = json.loads((args.output_dir / "summary.json").read_text())
            summary.update(image=str(args.image.resolve()), modelLoadSeconds=load_seconds,
                           generationSeconds=summary["totalSeconds"], totalSeconds=time.perf_counter() - started)
            write_json(args.output_dir / "summary.json", summary)
        return 0
    except Exception as error:
        LOGGER.exception("Probe failed; see full log")
        if not (args.output_dir / "summary.json").exists():
            write_json(args.output_dir / "summary.json", {
                "model": asdict(config), "seed": config.seed, "finalStatus": "failed",
                "error": str(error), "totalSeconds": time.perf_counter() - started,
            })
        if not (args.output_dir / "events.jsonl").exists():
            journal = EventJournal(args.output_dir, config.steps)
            journal.emit("task_failed", "failed", error=str(error))
            journal.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
