from enum import StrEnum

__all__ = (
    "CompressionEncoding",
    "DrainState",
    "HttpMethod",
    "MediaType",
    "OpenAPIMediaType",
    "ParamType",
    "RequestEncodingType",
    "ScopeType",
)


class HttpMethod(StrEnum):
    """An Enum for HTTP methods."""

    DELETE = "DELETE"
    GET = "GET"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    PATCH = "PATCH"
    POST = "POST"
    PUT = "PUT"
    TRACE = "TRACE"


class MediaType(StrEnum):
    """An Enum for ``Content-Type`` header values."""

    JSON = "application/json"
    MESSAGEPACK = "application/vnd.msgpack"
    HTML = "text/html"
    TEXT = "text/plain"
    CSS = "text/css"
    XML = "application/xml"


class OpenAPIMediaType(StrEnum):
    """An Enum for OpenAPI specific response ``Content-Type`` header values."""

    OPENAPI_YAML = "application/vnd.oai.openapi"
    OPENAPI_JSON = "application/vnd.oai.openapi+json"


class RequestEncodingType(StrEnum):
    """An Enum for request ``Content-Type`` header values designating encoding formats."""

    JSON = "application/json"
    MESSAGEPACK = "application/vnd.msgpack"
    MULTI_PART = "multipart/form-data"
    URL_ENCODED = "application/x-www-form-urlencoded"


class ScopeType(StrEnum):
    """An Enum for the 'http' key stored under Scope.

    Notes:
        - ``asgi`` is used by Litestar internally and is not part of the specification.
    """

    HTTP = "http"
    WEBSOCKET = "websocket"
    ASGI = "asgi"


class ParamType(StrEnum):
    """An Enum for the types of parameters a request can receive."""

    PATH = "path"
    QUERY = "query"
    COOKIE = "cookie"
    HEADER = "header"
    DEPENDENCY = "dependency"


class CompressionEncoding(StrEnum):
    """An Enum for supported compression encodings."""

    GZIP = "gzip"
    BROTLI = "br"
    ZSTD = "zstd"


class ASGIExtension(StrEnum):
    """ASGI extension keys: https://asgi.readthedocs.io/en/latest/extensions.html"""

    WS_DENIAL = "websocket.http.response"
    SERVER_PUSH = "http.response.push"
    ZERO_COPY_SEND_EXTENSION = "http.response.zerocopysend"
    PATH_SEND = "http.response.pathsend"
    TLS = "tls"
    EARLY_HINTS = "http.response.early_hint"
    HTTP_TRAILERS = "http.response.trailers"


class DrainState(StrEnum):
    """Observable state of an application's shutdown drain.

    ``RUNNING`` -> ``DRAINING`` -> ``DRAINED`` is the only legal transition order,
    and it is performed exactly once.
    """

    RUNNING = "running"
    """The application is serving traffic normally."""
    DRAINING = "draining"
    """A shutdown notification has been received. New requests are rejected, except
    configured probe paths, while in-flight requests are allowed to complete."""
    DRAINED = "drained"
    """The drain has completed - either because all in-flight requests finished or the
    configured deadline elapsed. Shutdown hooks run only after this state is reached."""
