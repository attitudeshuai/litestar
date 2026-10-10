from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from litestar.enums import CompressionEncoding
from litestar.exceptions import ImproperlyConfiguredException
from litestar.middleware.compression import CompressionMiddleware
from litestar.middleware.compression.gzip_facade import GzipCompression

if TYPE_CHECKING:
    from collections.abc import Sequence

    from litestar.middleware.compression.facade import CompressionFacade

    CompressionBackend = Literal["gzip", "brotli", "zstd"] | str | type[CompressionFacade]

__all__ = ("CompressionConfig",)


@dataclass
class CompressionConfig:
    """Configuration for response compression.

    To enable response compression, pass an instance of this class to the :class:`Litestar <.app.Litestar>` constructor
    using the ``compression_config`` key.
    """

    backend: Literal["gzip", "brotli", "zstd"] | str | None = None
    """The backend to use.

    If the value given is `gzip`, `brotli` or `zstd`, then the corresponding compression algorithm will be used.

    Required unless :attr:`backends` is provided.
    """
    minimum_size: int = field(default=500)
    """Minimum response size (bytes) to enable compression, affects all backends."""
    gzip_compress_level: int = field(default=9)
    """Range ``[0-9]``, see :doc:`python:library/gzip`."""
    zstd_compress_level: int = field(default=0)
    """Integer greater than or equal to 0.
    A value of 0 indicates use of default compression level set by the library.
    """
    brotli_quality: int = field(default=5)
    """Range ``[0-11]``, Controls the compression-speed vs compression-density tradeoff.

    The higher the quality, the slower the compression.
    """
    brotli_mode: Literal["generic", "text", "font"] = "text"
    """``MODE_GENERIC``, ``MODE_TEXT`` (for UTF-8 format text input, default) or ``MODE_FONT`` (for WOFF 2.0)."""
    brotli_lgwin: int = field(default=22)
    """Base 2 logarithm of size.

    Range is 10 to 24. Defaults to 22.
    """
    brotli_lgblock: Literal[0, 16, 17, 18, 19, 20, 21, 22, 23, 24] = 0
    """Base 2 logarithm of the maximum input block size.

    Range is ``16`` to ``24``. If set to ``0``, the value will be set based on the quality. Defaults to ``0``.
    """
    brotli_gzip_fallback: bool = True
    """Use GZIP if Brotli is not supported."""
    middleware_class: type[CompressionMiddleware] = CompressionMiddleware
    """Middleware class to use, should be a subclass of :class:`CompressionMiddleware`."""
    exclude: str | list[str] | None = None
    """A pattern or list of patterns to skip in the compression middleware."""
    exclude_opt_key: str | None = None
    """An identifier to use on routes to disable compression for a particular route."""
    compression_facade: type[CompressionFacade] = GzipCompression
    """The compression facade to use for the actual compression."""
    backend_config: Any = None
    """Configuration specific to the backend."""
    zstd_gzip_fallback: bool = True
    """Use GZIP as a fallback if Zstd is not supported by the client."""
    gzip_fallback: bool = True
    """Use GZIP as a fallback if the provided backend is not supported by the client."""
    backends: Sequence[CompressionBackend] | None = None
    """The content codings offered simultaneously to clients, in server preference order.

    Each entry is either the name of a built-in backend (``"gzip"``, ``"brotli"`` or
    ``"zstd"``) or a custom :class:`~litestar.middleware.compression.CompressionFacade`
    class. When more than one coding is offered, the coding returned to a client is
    selected from the request's ``Accept-Encoding`` header according to the client's
    quality (``q``) factors, with ties broken by the client's preference order. An
    absent ``Accept-Encoding`` header selects the first coding in the sequence.

    When ``backends`` is not provided, a single :attr:`backend` (optionally with
    :attr:`gzip_fallback`) is used and the legacy selection behaviour is preserved.
    """

    backend_facades: tuple[type[CompressionFacade], ...] = field(init=False)
    """The ordered compression facades offered to clients."""
    multi_backend: bool = field(init=False)
    """Whether multiple codings are offered simultaneously via :attr:`backends`."""

    def __post_init__(self) -> None:
        if self.minimum_size <= 0:
            raise ImproperlyConfiguredException("minimum_size must be greater than 0")

        if self.backends is not None:
            self.backends = tuple(self.backends)
            self._configure_backends()
            return

        if self.backend is None:
            raise ImproperlyConfiguredException("Either 'backend' or 'backends' must be provided")

        self._configure_single_backend()

    def _configure_single_backend(self) -> None:
        """Configure the legacy single-backend setup, preserving its exact behaviour."""
        self.multi_backend = False

        if self.backend == "gzip":
            if self.gzip_compress_level < 0 or self.gzip_compress_level > 9:
                raise ImproperlyConfiguredException("gzip_compress_level must be a value between 0 and 9")
        elif self.backend == "brotli":
            # Brotli is not guaranteed to be installed.
            from litestar.middleware.compression.brotli_facade import BrotliCompression

            self._validate_brotli_parameters()
            self.gzip_fallback = self.brotli_gzip_fallback
            self.compression_facade = BrotliCompression
        elif self.backend == "zstd":
            from litestar.middleware.compression.zstd_facade import ZstdCompression

            self._validate_zstd_level(ZstdCompression.upper_bound)
            self.gzip_fallback = self.zstd_gzip_fallback
            self.compression_facade = ZstdCompression

        facades = [self.compression_facade]
        if self.gzip_fallback and self.compression_facade.encoding != GzipCompression.encoding:
            facades.append(GzipCompression)
        self.backend_facades = tuple(facades)

    def _configure_backends(self) -> None:
        """Configure the multi-backend setup with RFC content negotiation."""
        if not self.backends:
            raise ImproperlyConfiguredException("'backends' must contain at least one backend")

        facades: list[type[CompressionFacade]] = []
        seen_encodings: set[str] = set()
        for configured_backend in self.backends:
            facade = self.resolve_facade(configured_backend)
            if facade.encoding in seen_encodings:
                continue
            seen_encodings.add(facade.encoding)
            facades.append(facade)

        for facade in facades:
            if facade is GzipCompression:
                if self.gzip_compress_level < 0 or self.gzip_compress_level > 9:
                    raise ImproperlyConfiguredException("gzip_compress_level must be a value between 0 and 9")
            elif (
                facade.__module__ == "litestar.middleware.compression.brotli_facade"
                and facade.__name__ == "BrotliCompression"
            ):
                self._validate_brotli_parameters()
            elif (
                facade.__module__ == "litestar.middleware.compression.zstd_facade"
                and facade.__name__ == "ZstdCompression"
            ):
                from litestar.middleware.compression.zstd_facade import ZstdCompression

                self._validate_zstd_level(ZstdCompression.upper_bound)

        self.multi_backend = True
        self.backend_facades = tuple(facades)
        # Keep the legacy attributes coherent for code consuming them directly.
        self.compression_facade = facades[0]
        self.backend = facades[0].encoding
        self.gzip_fallback = any(facade.encoding == GzipCompression.encoding for facade in facades)

    def _validate_brotli_parameters(self) -> None:
        if self.brotli_quality < 0 or self.brotli_quality > 11:
            raise ImproperlyConfiguredException("brotli_quality must be a value between 0 and 11")

        if self.brotli_lgwin < 10 or self.brotli_lgwin > 24:
            raise ImproperlyConfiguredException("brotli_lgwin must be a value between 10 and 24")

    def _validate_zstd_level(self, upper_bound: int) -> None:
        if not (0 <= self.zstd_compress_level <= upper_bound):
            raise ImproperlyConfiguredException(
                f"zstd_compress_level must be between 0 and {upper_bound}, given: {self.zstd_compress_level}"
            )

    @staticmethod
    def resolve_facade(backend: CompressionBackend) -> type[CompressionFacade]:
        """Resolve a backend name or facade class to a compression facade class.

        Args:
            backend: A built-in backend name or a compression facade class.

        Returns:
            The resolved compression facade class.

        Raises:
            ImproperlyConfiguredException: If the backend name is unknown.
        """
        if isinstance(backend, type):
            return backend

        if backend in ("gzip", CompressionEncoding.GZIP):
            return GzipCompression
        if backend in ("brotli", CompressionEncoding.BROTLI):
            # Brotli is not guaranteed to be installed.
            from litestar.middleware.compression.brotli_facade import BrotliCompression

            return BrotliCompression
        if backend == "zstd":
            from litestar.middleware.compression.zstd_facade import ZstdCompression

            return ZstdCompression

        raise ImproperlyConfiguredException(
            f"Unknown compression backend: {backend!r}. Expected one of 'gzip', 'brotli', 'zstd' or a "
            "CompressionFacade class."
        )
