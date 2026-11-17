#!/usr/bin/env python3
"""Opt-in real HTTP/WS acceptance check. Requires an already-running service; not a Java test."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import struct
import time
from uuid import uuid4

import httpx
from websockets.asyncio.client import connect


async def check(base_url, token, image, output_dir):
    headers = {"Authorization": "Bearer " + token}
    ws_base = base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    output_dir.mkdir(parents=True, exist_ok=False)
    received = []
    submit_times = []
    defaults = {"seed": 1234, "steps": 30, "guidanceScale": 5.0, "finalResolution": 256,
                "previewEnabled": True, "previewStep": 24, "previewResolution": 128,
                "pointCount": 5000, "numChunks": 4000}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30) as http:
        for attempt in range(180):
            response = await http.get("/readyz")
            if response.status_code == 200:
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Model did not become ready")

        async def submit(task_id, parameters):
            started = time.perf_counter()
            response = await http.post("/api/v1/tasks", data={"taskId": task_id, "parameters": json.dumps(parameters)},
                                       files={"image": (image.name, image.read_bytes(), "image/png")})
            submit_times.append(time.perf_counter() - started)
            return response

        first_id, second_id = str(uuid4()), str(uuid4())
        first = await submit(first_id, defaults)
        first.raise_for_status()
        assert first.status_code == 202 and submit_times[-1] < 5
        second = await submit(second_id, {**defaults, "previewEnabled": False})
        second.raise_for_status()
        second_initial = (await http.get(second.json()["snapshotPath"])).json()
        assert second_initial["status"] == "queued", second_initial

        async def receive(ws, previous=None):
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            assert event["protocolVersion"] == 1
            assert all(key in event for key in ("taskId", "sequence", "type", "stage", "completedSteps",
                                               "totalSteps", "previewVersion", "timestamp", "payload"))
            if previous is not None:
                assert event["sequence"] > previous
            received.append(event)
            return event

        events_url = ws_base + first.json()["eventsPath"]
        async with connect(events_url, additional_headers=headers, max_size=2**21) as ws:
            event = await receive(ws)
            assert event["type"] == "task_snapshot"
            previous = event["sequence"]
            while True:
                event = await receive(ws, previous)
                previous = event["sequence"]
                if event["type"] == "sampling_step" and event["completedSteps"] == 20:
                    break  # Deliberate disconnection; inference must continue independently.
        assert (await http.get("/healthz")).status_code == 200
        async with connect(events_url, additional_headers=headers, max_size=2**21) as ws:
            event = await receive(ws)
            assert event["type"] == "task_snapshot" and event["completedSteps"] >= 20
            previous = event["sequence"]
            while True:
                event = await receive(ws, previous)
                previous = event["sequence"]
                if event["type"] == "preview_ready":
                    points = event["payload"]["positions"]
                    assert event["payload"]["sourceStep"] == 24
                    assert len(points) == event["payload"]["pointCount"] == 5000
                    assert all(len(point) == 3 and all(math.isfinite(value) for value in point) for point in points)
                    (output_dir / "ws-preview.json").write_text(json.dumps(event, allow_nan=False))
                    break
                assert event["type"] not in ("task_failed", "task_completed", "preview_failed"), event
        # A second reconnect, after preview publication, restores the point cloud by HTTP.
        async with connect(events_url, additional_headers=headers, max_size=2**21) as ws:
            event = await receive(ws)
            snapshot = event["payload"]
            assert snapshot["latestPreview"]["version"] == 1
            cloud_response = await http.get(snapshot["latestPreview"]["positionsPath"])
            cloud_response.raise_for_status()
            assert cloud_response.json()["positions"] == points
            (output_dir / "http-preview.json").write_text(cloud_response.text)
            previous = event["sequence"]
            saw_finalizing = snapshot["stage"] == "final_decoding"
            while snapshot["status"] != "completed":
                event = await receive(ws, previous)
                previous = event["sequence"]
                if event["type"] == "finalizing":
                    saw_finalizing = True
                    assert event["stage"] == "final_decoding" and event["completedSteps"] == 30
                    state = (await http.get(first.json()["snapshotPath"])).json()
                    assert state["status"] == "running" and state["result"] is None
                    assert (await http.get(first.json()["snapshotPath"] + "/result")).status_code == 409
                assert event["type"] != "task_failed", event
                if event["type"] == "task_completed":
                    break
            assert saw_finalizing
        async with connect(events_url, additional_headers=headers, max_size=2**21) as ws:
            event = await receive(ws)
            assert event["type"] == "task_snapshot" and event["payload"]["status"] == "completed"
        replay = await submit(first_id, defaults)
        assert replay.status_code == 202 and replay.json()["status"] == "completed"
        conflict = await submit(first_id, {**defaults, "seed": 99})
        assert conflict.status_code == 409

        second_url = ws_base + second.json()["eventsPath"]
        async with connect(second_url, additional_headers=headers, max_size=2**21) as ws:
            event = await receive(ws)
            previous = event["sequence"]
            if event["payload"]["status"] != "completed":
                while True:
                    event = await receive(ws, previous)
                    previous = event["sequence"]
                    assert event["type"] != "task_failed", event
                    if event["type"] == "task_completed":
                        break
        states = []
        for task_id in (first_id, second_id):
            state = (await http.get(f"/api/v1/tasks/{task_id}")).json()
            states.append(state)
            model = await http.get(state["result"]["modelPath"])
            model.raise_for_status()
            assert model.headers["content-type"] == "model/gltf-binary"
            assert model.content[:4] == b"glTF" and struct.unpack_from("<I", model.content, 8)[0] == len(model.content)
            (output_dir / f"{task_id}.glb").write_bytes(model.content)
        summary = {"client": "Python HTTP/WS (not Java)", "taskIds": [first_id, second_id],
                   "submissionSeconds": submit_times, "secondInitiallyQueued": True,
                   "disconnectedAtStep": 20, "previewPointCount": 5000, "finalizingVerified": True,
                   "idempotentReplayVerified": True, "conflictingReplayStatus": 409, "snapshots": states}
        (output_dir / "received-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in received))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--image", type=Path, default=Path("assets/example_images/004.png"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("INFERENCE_API_TOKEN")
    if not token:
        parser.error("INFERENCE_API_TOKEN is required")
    asyncio.run(check(args.base_url.rstrip('/'), token, args.image, args.output_dir))


if __name__ == "__main__":
    main()
