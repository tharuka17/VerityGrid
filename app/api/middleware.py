"""Early rejection of oversized uploads.

FastAPI parses a multipart body before it calls a route, so a route cannot refuse a huge upload early.
Two checks run here, both before the route:

  1. The declared Content-Length of POST /scans: a value over the limit gets 413 before any body is read.
  2. A running count of the body bytes actually received. A client that omits Content-Length (chunked
     transfer) or states a false value is stopped as soon as the count passes the limit: the application
     is told the client disconnected, whatever it answers is discarded, and a 413 is sent instead. This
     bounds what the multipart parser can spool to memory or disk.

The route's bounded read and the ingestion limits still apply as a last line of defence. The limit applies
to the whole request body, so a small allowance is added for multipart framing.
"""

from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response
from app.core.config import get_settings

UPLOAD_PATH = "/scans"
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _declared_content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                length = int(value)
            except ValueError:
                return None
            return length if length >= 0 else None
    return None


def _is_upload(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope["method"] == "POST"
        and scope["path"].rstrip("/") == UPLOAD_PATH
    )


def _too_large(max_bytes: int) -> Response:
    return error_response(413, "file_too_large", f"The upload exceeds the limit of {max_bytes} bytes.")


class UploadSizeLimitMiddleware:
    """Pure ASGI middleware. It never buffers the body; it only counts the bytes that pass through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_upload(scope):
            await self.app(scope, receive, send)
            return

        max_bytes = get_settings().max_upload_bytes
        limit = max_bytes + MULTIPART_OVERHEAD_BYTES
        declared = _declared_content_length(scope)
        if declared is not None and declared > limit:
            await _too_large(max_bytes)(scope, receive, send)
            return

        received = 0
        exceeded = False
        started = False

        async def counted_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    return {"type": "http.disconnect"}  # nothing more of the body is read
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if exceeded and not started:
                return  # whatever the application answers is replaced by the 413 below
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except Exception:
            if not (exceeded and not started):
                raise  # an unrelated failure, or the response already began
        if exceeded and not started:
            await _too_large(max_bytes)(scope, receive, send)