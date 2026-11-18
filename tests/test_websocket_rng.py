import asyncio
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import uvicorn
from websockets.asyncio.client import connect
from websockets.frames import OP_PING
from websockets.legacy.protocol import WebSocketCommonProtocol

from hy3dgen.shapegen.preview import random_fingerprints
from inference_service.app import create_app
from inference_service.config import Settings
from inference_service.websocket_protocol import IsolatedHeartbeatProtocol
from test_inference_service import StubEngine, image_bytes


class WebSocketHeartbeatRngTests(unittest.IsolatedAsyncioTestCase):
    def protocol(self, protocol_type):
        # Exercise the installed ping implementation without a network or model.
        protocol = object.__new__(protocol_type)
        protocol.ensure_open = AsyncMock()
        protocol.write_frame = AsyncMock()
        protocol.pings = {}
        protocol.loop = asyncio.get_running_loop()
        return protocol

    def cleanup_waiters(self, protocol):
        for waiter, timestamp in protocol.pings.values():
            waiter.cancel()

    async def test_legacy_default_heartbeat_reproduces_global_python_rng_change(self):
        protocol = self.protocol(WebSocketCommonProtocol)
        before = random_fingerprints('cpu')
        await protocol.ping()
        after = random_fingerprints('cpu')
        self.assertNotEqual(before['python'], after['python'])
        self.cleanup_waiters(protocol)

    async def test_service_heartbeat_never_uses_global_random_and_keeps_all_rng_states(self):
        protocol = self.protocol(IsolatedHeartbeatProtocol)
        before = random_fingerprints('cpu')
        with patch('websockets.legacy.protocol.random.getrandbits', side_effect=AssertionError('global RNG used')):
            await protocol.ping()
            await protocol.ping()
        self.assertEqual(before, random_fingerprints('cpu'))
        self.assertEqual(protocol.write_frame.await_count, 2)
        self.assertEqual(len(protocol.pings), 2)
        self.cleanup_waiters(protocol)

    async def test_explicit_payload_and_duplicate_protection_are_preserved(self):
        protocol = self.protocol(IsolatedHeartbeatProtocol)
        await protocol.ping(b'explicit-payload')
        self.assertIn(b'explicit-payload', protocol.pings)
        with self.assertRaises(RuntimeError):
            await protocol.ping(b'explicit-payload')
        self.cleanup_waiters(protocol)

    async def test_real_ws_heartbeats_during_preview_do_not_fail_the_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(token='heartbeat-test', root=Path(temporary), device='cpu')
            engine = StubEngine()
            engine.pipeline.release()
            original_export = engine.pipeline._export

            def paused_export(**kwargs):
                time.sleep(.2)  # Several real transport heartbeats overlap the guarded preview.
                return original_export(**kwargs)

            app = create_app(settings, lambda: engine)
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            address = f'127.0.0.1:{sock.getsockname()[1]}'
            server = uvicorn.Server(uvicorn.Config(app, ws=IsolatedHeartbeatProtocol, log_level='error',
                                                   ws_ping_interval=.01, ws_ping_timeout=2))
            thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
            with patch.object(engine.pipeline, '_export', side_effect=paused_export):
                thread.start()
                try:
                    deadline = time.monotonic() + 5
                    while not server.started and time.monotonic() < deadline:
                        await asyncio.sleep(.01)
                    self.assertTrue(server.started)
                    self.assertTrue(await asyncio.to_thread(app.state.manager.loaded.wait, 2))
                    task_id = str(uuid4())
                    headers = {'Authorization': 'Bearer heartbeat-test'}
                    async with httpx.AsyncClient(base_url='http://' + address, headers=headers) as client:
                        response = await client.post('/api/v1/tasks', data={'taskId': task_id},
                                                     files={'image': ('input.png', image_bytes(), 'image/png')})
                        self.assertEqual(response.status_code, 202)
                        async with connect('ws://' + address + response.json()['eventsPath'],
                                           additional_headers=headers, ping_interval=None) as ws:
                            pings = []
                            original_frame = ws.protocol.recv_frame

                            def receive_frame(frame):
                                if frame.opcode == OP_PING:
                                    pings.append(time.monotonic())
                                return original_frame(frame)

                            ws.protocol.recv_frame = receive_frame
                            self.assertEqual(json.loads(await ws.recv())['type'], 'task_snapshot')
                            engine.start_gate.set()
                            preview_ping_count = 0
                            while True:
                                event = json.loads(await asyncio.wait_for(ws.recv(), 5))
                                self.assertNotIn(event['type'], ('preview_failed', 'task_failed'))
                                if event['type'] == 'preview_started':
                                    preview_ping_count = len(pings)
                                elif event['type'] == 'preview_ready':
                                    self.assertGreater(len(pings), preview_ping_count)
                                elif event['type'] == 'task_completed':
                                    break
                        summary = json.loads((settings.root / task_id / 'summary.json').read_text())
                        self.assertTrue(summary['preview']['randomStateUnchanged'])
                finally:
                    engine.release()
                    server.should_exit = True
                    await asyncio.to_thread(thread.join, 5)
                    sock.close()


if __name__ == '__main__':
    unittest.main()
