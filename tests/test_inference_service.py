import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import torch

from inference_service.app import create_app
from inference_service.config import Settings
from inference_service.engine import ServiceJournal
from inference_service.parameters import TaskParameters
from inference_service.store import TaskStore
from inference_service.subscriptions import SlowSubscriber, Subscription
from hy3dgen.shapegen.preview import PreviewProbe, ProbeConfig
from test_preview_probe import FakePipeline, tetrahedron


def image_bytes():
    output = io.BytesIO()
    Image.new('RGB', (16, 16), 'orange').save(output, format='PNG')
    return output.getvalue()


class GatedPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.at20 = threading.Event()
        self.continue20 = threading.Event()
        self.at24 = threading.Event()
        self.continue24 = threading.Event()
        self.finalizing = threading.Event()
        self.continue_final = threading.Event()

    def __call__(self, **kwargs):
        from types import SimpleNamespace
        for index in range(kwargs['num_inference_steps']):
            self.latent = torch.full((1, 2, 3), float(index + 1))
            kwargs['callback'](index, torch.tensor(index), SimpleNamespace(prev_sample=self.latent))
            if index == 19:
                self.at20.set()
                self.continue20.wait(10)
            if index == 23:
                self.at24.set()
                self.continue24.wait(10)
        self.finalizing.set()
        self.continue_final.wait(10)
        return [tetrahedron()]

    def release(self):
        self.continue20.set()
        self.continue24.set()
        self.continue_final.set()


class StubEngine:
    """Protocol-only fixture: real shared preview code, synthetic tetrahedron decoder, no weights."""
    def __init__(self):
        self.start_gate = threading.Event()
        self.pipeline = GatedPipeline()
        self.calls = []
        self.fail_preview = False
        self.device_failure = False

    def run(self, task, store):
        self.calls.append(task.task_id)
        self.start_gate.wait(10)
        parameters = task.parameters
        config = ProbeConfig(device='cpu', steps=parameters['steps'], preview_step=parameters['previewStep'],
                             preview_stages=tuple(parameters.get('previewStages') or ()))
        probe = PreviewProbe(config, store.directory(task.task_id), enabled=parameters['previewEnabled'],
                             journal=ServiceJournal(store, task))
        if self.fail_preview or self.device_failure:
            failure = torch.OutOfMemoryError('test device failure') if self.device_failure else ValueError('private internal path /secret')
            with patch.object(self.pipeline, '_export', side_effect=failure):
                probe.run(self.pipeline, None)
        else:
            probe.run(self.pipeline, None)

    def release(self):
        self.start_gate.set()
        self.pipeline.release()


class ServiceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(token='local-test-token', root=Path(self.temporary.name), device='cpu', queue_capacity=1)
        self.engine = StubEngine()
        self.app = create_app(self.settings, lambda: self.engine)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(self.engine.release)
        self.assertTrue(self.app.state.manager.loaded.wait(2))
        self.headers = {'Authorization': 'Bearer ' + self.settings.token}

    def submit(self, task_id=None, parameters=None, content=None):
        return self.client.post('/api/v1/tasks', headers=self.headers,
                                data={'taskId': task_id or str(uuid4()), 'parameters': json.dumps(parameters or {})},
                                files={'image': ('source.png', content or image_bytes(), 'image/png')})

    def state(self, task_id):
        return self.client.get('/api/v1/tasks/' + task_id, headers=self.headers).json()

    def wait_completed(self, task_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.state(task_id)
            if state['status'] in ('completed', 'failed'):
                self.assertEqual(state['status'], 'completed', state)
                return state
            time.sleep(0.01)
        self.fail('Task did not complete')

    def test_legacy_persisted_request_fingerprint_survives_optional_plan_addition(self):
        task_id = str(uuid4())
        legacy = TaskParameters().model_dump(exclude_none=True)
        self.assertNotIn('previewStages', legacy)
        self.app.state.manager.store.create(task_id, legacy, image_bytes())
        response = self.submit(task_id=task_id)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()['taskId'], task_id)
        self.assertEqual(self.engine.calls, [])

    def test_all_three_preview_versions_remain_readable_after_final(self):
        stages = [{'step': 6, 'resolution': 64, 'pointCount': 1000},
                  {'step': 15, 'resolution': 96, 'pointCount': 2500},
                  {'step': 24, 'resolution': 128, 'pointCount': 5000}]
        response = self.submit(parameters={'previewStages': stages})
        self.assertEqual(response.status_code, 202)
        task_id = response.json()['taskId']
        self.engine.release()
        state = self.wait_completed(task_id)
        self.assertEqual(state['previewVersion'], 3)
        for version, stage in enumerate(stages, 1):
            result = self.client.get(f'/api/v1/tasks/{task_id}/previews/{version}', headers=self.headers)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['sourceStep'], stage['step'])
            self.assertEqual(len(result.json()['positions']), stage['pointCount'])
        self.assertEqual(self.client.get(f'/api/v1/tasks/{task_id}/previews/4', headers=self.headers).status_code, 404)

    def test_rejects_unordered_duplicate_or_oversized_preview_plans(self):
        for stages in ([{'step': 15, 'resolution': 96, 'pointCount': 2500}, {'step': 6, 'resolution': 64, 'pointCount': 1000}],
                       [{'step': 30, 'resolution': 128, 'pointCount': 5000}],
                       [{'step': 6, 'resolution': 256, 'pointCount': 1000}],
                       [{'step': 6, 'resolution': 64, 'pointCount': 5001}]):
            self.assertEqual(self.submit(parameters={'previewStages': stages}).status_code, 422)

    def test_snapshot_progress_disconnect_preview_recovery_finalizing_and_terminal_subscription(self):
        started = time.monotonic()
        response = self.submit()
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()['status'], 'queued')
        task_id = response.json()['taskId']
        path = response.json()['snapshotPath']
        events_path = response.json()['eventsPath']
        with self.client.websocket_connect(events_path, headers=self.headers) as ws:
            first = ws.receive_json()
            self.assertEqual(first['type'], 'task_snapshot')
            self.engine.start_gate.set()
            while True:
                event = ws.receive_json()
                self.assertGreater(event['sequence'], first['sequence'])
                if event['type'] == 'sampling_step' and event['completedSteps'] == 20:
                    break
        self.assertTrue(self.engine.pipeline.at20.wait(2))
        self.assertEqual(self.state(task_id)['completedSteps'], 20)
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.engine.pipeline.continue20.set()
        self.assertTrue(self.engine.pipeline.at24.wait(2))
        with self.client.websocket_connect(events_path, headers=self.headers) as ws:
            control = ws.receive_json()
            self.assertEqual(control['type'], 'task_snapshot')
            latest = control['payload']['latestPreview']
            self.assertEqual(latest['version'], 1)
            preview = self.client.get(latest['positionsPath'], headers=self.headers).json()
            self.assertEqual(len(preview['positions']), 5000)
            self.assertTrue(np.isfinite(preview['positions']).all())
            self.assertEqual(preview['sourceStep'], 24)
            self.engine.pipeline.continue24.set()
            while (event := ws.receive_json())['type'] != 'finalizing':
                pass
            state = self.state(task_id)
            self.assertEqual(state['stage'], 'final_decoding')
            self.assertEqual(state['completedSteps'], 30)
            self.assertEqual(state['status'], 'running')
            self.assertEqual(self.client.get(path + '/result', headers=self.headers).status_code, 409)
            self.engine.pipeline.continue_final.set()
            while (event := ws.receive_json())['type'] != 'task_completed':
                pass
            model = self.client.get(event['payload']['modelPath'], headers=self.headers)
            self.assertEqual(model.headers['content-type'], 'model/gltf-binary')
            self.assertEqual(model.content[:4], b'glTF')
        with self.client.websocket_connect(events_path, headers=self.headers) as ws:
            control = ws.receive_json()
            self.assertEqual(control['payload']['status'], 'completed')
            self.assertEqual(control['sequence'], self.state(task_id)['sequence'])
        replay = self.submit(task_id, TaskParameters().model_dump())
        self.assertEqual(replay.status_code, 202)
        self.assertEqual(self.engine.calls, [task_id])
        self.assertEqual(self.submit(task_id, {'seed': 99}).status_code, 409)
        persisted = [json.loads(line) for line in (self.settings.root / task_id / 'events.jsonl').read_text().splitlines()]
        ready = next(event for event in persisted if event['type'] == 'preview_ready')
        self.assertEqual(len(ready['payload']['positions']), 5000)
        self.assertFalse(any('/secret' in json.dumps(event) for event in persisted))

    def test_atomic_concurrent_duplicates_bounded_queue_and_serial_inference(self):
        task_id = str(uuid4())
        with ThreadPoolExecutor(max_workers=6) as pool:
            replies = list(pool.map(lambda _: self.submit(task_id).status_code, range(6)))
        self.assertEqual(replies, [202] * 6)
        deadline = time.monotonic() + 2
        while not self.engine.calls and time.monotonic() < deadline:
            time.sleep(.01)
        second = self.submit()
        self.assertEqual(second.status_code, 202)
        self.assertEqual(self.submit().status_code, 429)
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.engine.release()
        self.wait_completed(task_id)
        self.wait_completed(second.json()['taskId'])
        self.assertEqual(self.engine.calls, [task_id, second.json()['taskId']])

    def test_auth_validation_and_error_responses_do_not_expose_details(self):
        self.assertEqual(self.client.get('/healthz').status_code, 200)
        self.assertEqual(self.client.get('/api/v1/tasks/' + str(uuid4())).status_code, 401)
        self.assertEqual(self.client.get('/api/v1/tasks/' + str(uuid4()), headers=self.headers).status_code, 404)
        for parameters in ({'steps': 1}, {'previewStep': 30}, {'pointCount': 5001}, {'unknown': 1}, {'guidanceScale': float('nan')}):
            self.assertEqual(self.submit(parameters=parameters).status_code, 422)
        self.assertEqual(self.submit(content=b'not an image').status_code, 422)
        self.assertEqual(self.submit(content=b'x' * (11 * 1024 * 1024)).status_code, 413)
        from starlette.websockets import WebSocketDisconnect
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/api/v1/tasks/' + str(uuid4()) + '/events'):
                pass

    def test_preview_failure_is_nonfatal_and_private_errors_are_sanitized(self):
        self.engine.fail_preview = True
        response = self.submit()
        task_id = response.json()['taskId']
        self.engine.release()
        self.wait_completed(task_id)
        events = [json.loads(line) for line in (self.settings.root / task_id / 'events.jsonl').read_text().splitlines()]
        failure = next(event for event in events if event['type'] == 'preview_failed')
        self.assertEqual(failure['payload']['code'], 'PREVIEW_DECODE_FAILED')
        self.assertFalse(failure['payload']['fatal'])
        self.assertTrue(all(event['previewVersion'] == 0 for event in events))
        self.assertNotIn('/secret', json.dumps(events))

    def test_slow_websocket_is_closed_1013_without_blocking_task(self):
        from starlette.websockets import WebSocket, WebSocketDisconnect
        original = WebSocket.send_json

        async def delayed(websocket, data, mode='text'):
            if data['type'] == 'sampling_step':
                await asyncio.sleep(.15)
            await original(websocket, data, mode=mode)

        manager = self.app.state.manager
        manager.store.settings = replace(self.settings, subscriber_capacity=1)
        response = self.submit()
        task_id = response.json()['taskId']
        with patch.object(WebSocket, 'send_json', new=delayed):
            with self.client.websocket_connect(response.json()['eventsPath'], headers=self.headers) as ws:
                self.assertEqual(ws.receive_json()['type'], 'task_snapshot')
                self.engine.release()
                with self.assertRaises(WebSocketDisconnect) as disconnected:
                    while True:
                        ws.receive_json()
                self.assertEqual(disconnected.exception.code, 1013)
        self.wait_completed(task_id)

    def test_device_failure_disables_worker_and_fails_waiting_task(self):
        self.engine.device_failure = True
        first = self.submit().json()['taskId']
        deadline = time.monotonic() + 2
        while not self.engine.calls and time.monotonic() < deadline:
            time.sleep(.01)
        second = self.submit().json()['taskId']
        self.engine.release()
        self.assertTrue(self.app.state.manager.failed.wait(3))
        self.assertEqual(self.client.get('/readyz').status_code, 503)
        self.assertEqual(self.state(first)['status'], 'failed')
        self.assertEqual(self.state(second)['error']['code'], 'WORKER_UNAVAILABLE')
        self.assertEqual(self.submit().status_code, 503)


class PersistenceAndSubscriptionTests(unittest.TestCase):
    def test_restart_fails_interrupted_tasks_preserves_preview_and_completed_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(token='test', root=Path(temporary), device='cpu')
            store = TaskStore(settings)
            store.initialize()
            pending, running, completed = (str(uuid4()) for _ in range(3))
            parameters = TaskParameters().model_dump()
            for task_id in (pending, running, completed):
                store.create(task_id, parameters, image_bytes())
            store.emit(running, 'task_started', 'preprocessing')
            (store.directory(running) / 'preview_step_024.json').write_text(json.dumps({'positions': [[1., 2., 3.]]}))
            store.emit(running, 'preview_ready', 'sampling', completed_steps=24, preview_version=1, payload={
                'sourceStep': 24, 'pointCount': 1, 'positions': [[1., 2., 3.]],
                'positionsPath': f'/api/v1/tasks/{running}/previews/1', 'decodeSeconds': 1,
            })
            tetrahedron().export(store.directory(completed) / 'final.glb')
            store.emit(completed, 'task_completed', 'completed', completed_steps=30,
                       payload={'modelPath': f'/api/v1/tasks/{completed}/result', 'format': 'glb'})
            with (store.directory(running) / 'events.jsonl').open('ab') as stream:
                stream.write(b'{"torn":')
            store.close()
            engine = StubEngine()
            engine.release()
            with TestClient(create_app(settings, lambda: engine)) as client:
                headers = {'Authorization': 'Bearer test'}
                for task_id in (pending, running):
                    state = client.get(f'/api/v1/tasks/{task_id}', headers=headers).json()
                    self.assertEqual(state['status'], 'failed')
                    self.assertEqual(state['error']['code'], 'SERVICE_RESTARTED')
                self.assertEqual(client.get(f'/api/v1/tasks/{running}/previews/1', headers=headers).status_code, 200)
                self.assertEqual(client.get(f'/api/v1/tasks/{completed}/result', headers=headers).content[:4], b'glTF')
                self.assertEqual(engine.calls, [])

    def test_slow_subscription_has_bounded_memory_and_reports_overflow(self):
        async def check():
            subscription = Subscription(2, asyncio.get_running_loop())
            for sequence in range(1000):
                subscription.offer({'sequence': sequence})
            self.assertLessEqual(len(subscription.items), 2)
            with self.assertRaises(SlowSubscriber):
                await subscription.next()
            subscription.close()
        asyncio.run(check())

    def test_missing_token_fails_before_model_loading(self):
        with self.assertRaises(ValueError):
            Settings(token='', root=Path('/tmp/unused'))

    def test_initializing_health_is_live_but_readiness_unavailable_and_store_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(token='test', root=Path(temporary), device='cpu')
            gate = threading.Event()
            engine = StubEngine()
            engine.release()

            def load():
                gate.wait(5)
                return engine

            app = create_app(settings, load)
            with TestClient(app) as client:
                try:
                    self.assertEqual(client.get('/healthz').status_code, 200)
                    self.assertEqual(client.get('/readyz').status_code, 503)
                    with self.assertRaises(RuntimeError):
                        other = TaskStore(settings)
                        other.initialize()
                    reply = client.post('/api/v1/tasks', headers={'Authorization': 'Bearer test'},
                                        data={'taskId': str(uuid4())}, files={'image': ('a.png', image_bytes(), 'image/png')})
                    self.assertEqual(reply.status_code, 503)
                finally:
                    gate.set()
                self.assertTrue(app.state.manager.loaded.wait(2))


if __name__ == '__main__':
    unittest.main()
