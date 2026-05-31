"""The value every tool returns.

:class:`ToolResult` is the single shape a tool handler may produce. It carries
the text the model will read, an error flag, and optional metadata for hooks
and the transcript. It also owns *layer 1* of the SPEC §5.4 tool-result defense:
per-tool character truncation. Layers 2 (spill-to-disk) and 3 (per-turn
aggregate budget) live elsewhere and are deferred to Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """The outcome of a single tool call, normalized for the model and loop.

    Attributes:
        content: The text the model sees as the tool's observation.
        is_error: ``True`` when the call failed; the loop still feeds the
            content back to the model so it can react (errors are not retried).
        metadata: Free-form structured data attached by the tool or hooks
            (e.g. byte counts, spill paths). Never shown to the model directly;
            it rides alongside the result for observability.
    """

    content: str
    is_error: bool = False
    metadata: dict = field(default_factory=dict)

    @classmethod
    def text(cls, content: str, **metadata: object) -> ToolResult:
        """Build a successful result from ``content``.

        Args:
            content: The observation text for the model.
            **metadata: Optional key/value pairs stored on :attr:`metadata`.

        Returns:
            A :class:`ToolResult` with ``is_error`` unset.
        """
        return cls(content=content, is_error=False, metadata=dict(metadata))

    @classmethod
    def error(cls, message: str, **metadata: object) -> ToolResult:
        """Build a failed result from ``message``.

        Args:
            message: The error text for the model to read and react to.
            **metadata: Optional key/value pairs stored on :attr:`metadata`.

        Returns:
            A :class:`ToolResult` with ``is_error`` set to ``True``.
        """
        return cls(content=message, is_error=True, metadata=dict(metadata))

    def truncate(self, max_chars: int) -> ToolResult:
        """Return a result whose content fits within ``max_chars`` (SPEC §5.4 layer 1).

        When :attr:`content` is at or under ``max_chars`` the receiver is
        returned unchanged. Otherwise a *new* result is produced whose content
        is a head slice and a tail slice of the original — the budget split
        roughly in half — joined by a marker stating how many characters were
        elided. The head/tail framing preserves the most useful context (the
        start and end of long output) while bounding what the model must read.

        Spill-to-disk (layer 2) would let the full output remain retrievable
        via a file path; it is deferred to Phase 3 and not implemented here.

        Args:
            max_chars: Maximum number of characters of original content to
                keep across the head and tail slices combined.

        Returns:
            ``self`` if under the cap, else a new truncated :class:`ToolResult`
            carrying the same ``is_error`` and a shallow copy of ``metadata``.
        """
        if len(self.content) <= max_chars:
            return self

        elided = len(self.content) - max_chars
        head_len = max_chars // 2
        tail_len = max_chars - head_len
        head = self.content[:head_len]
        tail = self.content[len(self.content) - tail_len :] if tail_len else ""
        marker = f"\n…[truncated {elided} chars]…\n"
        return ToolResult(
            content=head + marker + tail,
            is_error=self.is_error,
            metadata=dict(self.metadata),
        )
