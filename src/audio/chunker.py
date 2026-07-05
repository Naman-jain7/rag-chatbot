"""
src/audio/chunker.py
--------------------
Segment-grouping chunker for Whisper transcripts.

Strategy
--------
Group consecutive Whisper segments into a chunk until the accumulated
character count reaches ``chunk_size``.  Then slide back ``chunk_overlap``
characters worth of segments to form the next chunk's seed, preserving
natural speech-pause boundaries exactly as Whisper detected them.

Each output chunk is a dict::

    {
        "chunk_text":  "..transcript text for this chunk..",
        "start_time":  0.0,      # start of first segment in chunk (seconds)
        "end_time":    30.4,     # end   of last  segment in chunk (seconds)
        "chunk_index": 0,
    }
"""

from __future__ import annotations

from typing import Any


def _format_ts(seconds: float) -> str:
    """Format a float seconds value as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


class AudioChunker:
    """
    Groups Whisper timestamp segments into overlapping text chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk (default 1000).
    chunk_overlap : int
        Number of characters of overlap between consecutive chunks (default 200).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, timestamps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Split a list of Whisper segments into overlapping text chunks.

        Parameters
        ----------
        timestamps : list[dict]
            Each dict must have keys ``start`` (float), ``end`` (float),
            ``text`` (str).  This is the ``metadata.timestamps`` list
            returned by :class:`~src.audio.transcriber.AudioTranscriber`.

        Returns
        -------
        list[dict]
            Ordered list of chunk dicts, each with ``chunk_text``,
            ``start_time``, ``end_time``, ``chunk_index``.
        """
        if not timestamps:
            return []

        chunks: list[dict[str, Any]] = []
        # Seed with the first segment
        current_segs: list[dict[str, Any]] = []
        current_len: int = 0
        chunk_index: int = 0

        for seg in timestamps:
            seg_text = seg["text"]
            seg_len = len(seg_text)

            # If this single segment already exceeds chunk_size, emit it alone
            if not current_segs and seg_len >= self.chunk_size:
                chunks.append(self._make_chunk(chunk_index, [seg]))
                chunk_index += 1
                current_segs = []
                current_len = 0
                continue

            # Would adding this segment exceed the budget?
            # (+1 for the space separator between segments)
            sep = 1 if current_segs else 0
            if current_len + sep + seg_len > self.chunk_size and current_segs:
                # Emit the current chunk
                chunks.append(self._make_chunk(chunk_index, current_segs))
                chunk_index += 1

                # Slide back: keep trailing segments that fit within chunk_overlap
                current_segs = self._overlap_tail(current_segs)
                current_len = sum(len(s["text"]) for s in current_segs) + max(len(current_segs) - 1, 0)

            current_segs.append(seg)
            sep = 1 if len(current_segs) > 1 else 0
            current_len += sep + seg_len

        # Emit any remaining segments
        if current_segs:
            chunks.append(self._make_chunk(chunk_index, current_segs))

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self, index: int, segs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text = " ".join(s["text"] for s in segs)
        return {
            "chunk_text":  text,
            "start_time":  segs[0]["start"],
            "end_time":    segs[-1]["end"],
            "chunk_index": index,
            # Human-readable timestamp range stored as page_range in DB
            "page_range":  f"{_format_ts(segs[0]['start'])}–{_format_ts(segs[-1]['end'])}",
        }

    def _overlap_tail(
        self, segs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Return the trailing sub-list of *segs* whose total character count
        fits within ``self.chunk_overlap``.
        """
        tail: list[dict[str, Any]] = []
        budget = self.chunk_overlap
        for seg in reversed(segs):
            seg_len = len(seg["text"])
            if budget - seg_len < 0:
                break
            tail.insert(0, seg)
            budget -= seg_len + 1  # +1 for separator

        return tail


# Module-level singleton with defaults matching the existing doc splitter
audio_chunker = AudioChunker(chunk_size=1000, chunk_overlap=200)
