from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = (
    "WILDCARD",
    "AcceptedCoding",
    "ContentCodingSelection",
    "NegotiationResult",
    "parse_accept_encoding",
    "select_content_coding",
)

WILDCARD: Final[str] = "*"
IDENTITY: Final[str] = "identity"


class NegotiationResult(StrEnum):
    """The outcome of an ``Accept-Encoding`` negotiation."""

    ENCODE = "encode"
    """A content coding was negotiated and the response should be encoded with it."""
    IDENTITY = "identity"
    """No coding was negotiated; the response must be sent without a content coding."""
    NOT_ACCEPTABLE = "not_acceptable"
    """Neither an offered coding nor ``identity`` is acceptable to the client."""


@dataclass(frozen=True)
class AcceptedCoding:
    """A single entry parsed from an ``Accept-Encoding`` header."""

    coding: str
    """The content coding token (lower-cased), or ``*`` for the wildcard."""
    quality: float
    """The relative quality (``q``) factor, in the ``[0, 1]`` range. A value of ``0``
    means the client explicitly rejects the coding."""
    position: int
    """The position of the entry within the header, used to break quality ties in
    the client's preference order."""


@dataclass(frozen=True)
class ContentCodingSelection:
    """The result of negotiating a content coding for a response."""

    result: NegotiationResult
    """The negotiation outcome."""
    coding: str | None = None
    """The selected content coding when :attr:`result <NegotiationResult.ENCODE>`,
    otherwise ``None``."""


def parse_accept_encoding(accept_encoding: str | None) -> tuple[AcceptedCoding, ...]:
    """Parse an ``Accept-Encoding`` header into ordered ``(coding, quality)`` entries.

    See `RFC 9110, section 12.5.3 <https://www.rfc-editor.org/rfc/rfc9110#section-12.5.3>`_.

    Args:
        accept_encoding: The raw header value. ``None`` or an empty string denotes an
            absent header, meaning any content coding is acceptable.

    Returns:
        A tuple of :class:`AcceptedCoding` in header order. An empty tuple means the
        header was absent.
    """
    if accept_encoding is None:
        return ()

    entries: list[AcceptedCoding] = []
    for position, raw_value in enumerate(accept_encoding.split(",")):
        value = raw_value.strip()
        if not value:
            continue
        token, _, parameters = value.partition(";")
        coding = token.strip().lower()
        if not coding:
            continue

        quality = 1.0
        for raw_parameter in parameters.split(";"):
            parameter = raw_parameter.strip()
            if not parameter:
                continue
            name, separator, value_ = parameter.partition("=")
            if separator and name.strip().lower() == "q":
                try:
                    parsed_quality = float(value_.strip())
                except ValueError:
                    # An invalid quality value means the coding is not acceptable.
                    quality = 0.0
                    break
                if not 0 <= parsed_quality <= 1:
                    quality = 0.0
                    break
                quality = parsed_quality

        entries.append(AcceptedCoding(coding=coding, quality=quality, position=position))

    return tuple(entries)


def select_content_coding(
    offered: Sequence[str],
    accept_encoding: str | None,
    *,
    select_on_absent_header: bool,
    client_order_tie_break: bool,
) -> ContentCodingSelection:
    """Select a content coding from ``offered`` based on an ``Accept-Encoding`` header.

    Codings assigned a quality of ``0`` are treated as explicitly rejected by the
    client. If none of the offered codings is acceptable but ``identity`` is, the
    response is sent uncompressed (:attr:`~NegotiationResult.IDENTITY`). If even
    ``identity`` is rejected, :attr:`~NegotiationResult.NOT_ACCEPTABLE` is returned
    instead of silently falling back to an unsupported coding.

    Args:
        offered: The content codings supported by the server, in server preference
            order.
        accept_encoding: The raw ``Accept-Encoding`` request header value.
        select_on_absent_header: When the header is absent/empty and this is ``True``,
            the first offered coding is selected (per RFC); when ``False``, an absent
            header results in :attr:`~NegotiationResult.IDENTITY`, preserving the
            legacy behaviour.
        client_order_tie_break: When multiple acceptable codings share the highest
            quality, break the tie using the client's header order when ``True`` (per
            RFC) or the server preference order when ``False`` (legacy behaviour).

    Returns:
        A :class:`ContentCodingSelection` describing the negotiated coding.
    """
    entries = parse_accept_encoding(accept_encoding)

    if not entries:
        # An absent Accept-Encoding header means any coding is acceptable per RFC.
        if select_on_absent_header and offered:
            return ContentCodingSelection(result=NegotiationResult.ENCODE, coding=offered[0])
        return ContentCodingSelection(result=NegotiationResult.IDENTITY)

    explicit: dict[str, AcceptedCoding] = {}
    wildcard: AcceptedCoding | None = None
    for entry in entries:
        if entry.coding == WILDCARD:
            if wildcard is None:
                wildcard = entry
        else:
            explicit.setdefault(entry.coding, entry)

    # ``identity`` is always acceptable unless it (or everything via ``*``) is
    # explicitly assigned a quality of 0.
    identity_entry = explicit.get(IDENTITY)
    identity_quality = (
        identity_entry.quality if identity_entry is not None else wildcard.quality if wildcard is not None else 1.0
    )

    wildcard_precedence = len(entries) + 1_000_000
    candidates: list[tuple[float, int, int, str]] = []
    for server_index, coding in enumerate(offered):
        accepted_coding = explicit.get(coding)
        if accepted_coding is not None:
            quality = accepted_coding.quality
            precedence = accepted_coding.position if client_order_tie_break else server_index
        elif wildcard is not None:
            quality = wildcard.quality
            precedence = wildcard_precedence + server_index if client_order_tie_break else server_index
        else:
            quality = 0.0
            precedence = server_index

        if quality > 0:
            candidates.append((quality, precedence, server_index, coding))

    if candidates:
        # Highest quality first; ties resolved by precedence, then server order.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return ContentCodingSelection(result=NegotiationResult.ENCODE, coding=candidates[0][3])

    if identity_quality > 0:
        return ContentCodingSelection(result=NegotiationResult.IDENTITY)

    return ContentCodingSelection(result=NegotiationResult.NOT_ACCEPTABLE)
