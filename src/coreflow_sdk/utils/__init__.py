from .audit import AppLogger
from .env import ENV

__all__ = ["AppLogger", "ENV", "StreamingResponse", "create_streaming_response"]


def __getattr__(name):
    if name == "AppLogger":
        from .audit import AppLogger

        return AppLogger
    elif name == "ENV":
        from .env import ENV

        return ENV
    elif name == "StreamingResponse":
        from .streaming import StreamingResponse

        return StreamingResponse
    elif name == "create_streaming_response":
        from .streaming import create_streaming_response

        return create_streaming_response
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
