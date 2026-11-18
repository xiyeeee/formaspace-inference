"""Keep transport heartbeats independent of the inference RNG audit."""
import secrets

from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol


class IsolatedHeartbeatProtocol(WebSocketProtocol):
    async def ping(self, data=None):
        # websockets 15 legacy ping() otherwise calls random.getrandbits() on the
        # process-global Python generator, including while the GPU preview runs.
        if data is None:
            data = secrets.token_bytes(16)
            while data in self.pings:
                data = secrets.token_bytes(16)
        return await super().ping(data)
