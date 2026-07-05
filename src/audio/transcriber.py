"""
src/audio/transcriber.py
------------------------
Thin wrapper around faster-whisper that returns a canonical transcript dict:

    {
        "text": "..full transcript..",
        "metadata": {
            "source":     "meeting.mp3",
            "duration":   325.4,          # total audio length in seconds
            "timestamps": [               # one entry per whisper segment
                {"start": 0.0, "end": 4.2, "text": "Hello everyone."},
                ...
            ]
        }
    }

The transcriber is intentionally side-effect-free (no DB, no chunking).
Call it inside asyncio.to_thread() from the async upload endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.config import settings
from src.utils.logger import APP_LOGGER


def _resolve_device(device_setting: str) -> str:
    """Resolve 'auto' to 'cuda' (if available) or 'cpu'."""
    if device_setting != "auto":
        return device_setting
    try:
        import torch  # type: ignore[import-untyped]
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class AudioTranscriber:
    """
    Lazy-initialised faster-whisper transcriber.

    The WhisperModel is loaded only on the first call to ``transcribe()``
    so it does not block application startup.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def _get_model(self) -> Any:
        """Return (and lazily initialise) the WhisperModel."""
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            model_size = settings.whisper.WHISPER_MODEL_SIZE
            compute_type = settings.whisper.WHISPER_COMPUTE_TYPE
            device = _resolve_device(settings.whisper.WHISPER_DEVICE)

            APP_LOGGER.info(
                f"Loading faster-whisper model '{model_size}' "
                f"on device='{device}' compute_type='{compute_type}'"
            )
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            APP_LOGGER.info("faster-whisper model loaded successfully.")

        return self._model

    def transcribe(self, file_path: str) -> dict[str, Any]:
        """
        Transcribe an audio file synchronously.

        Parameters
        ----------
        file_path : str
            Absolute path to the audio file (MP3, WAV, M4A, OGG, FLAC …).

        Returns
        -------
        dict
            Canonical transcript dict with keys ``text`` and ``metadata``.
        """
        filename = os.path.basename(file_path)
        APP_LOGGER.info(f"Starting transcription for '{filename}'")

        model = self._get_model()

        # faster-whisper returns (segments_generator, TranscriptionInfo)
        segments_gen, info = model.transcribe(
            file_path,
            beam_size=5,
            word_timestamps=False,  # segment-level timestamps are sufficient
        )

        duration: float = info.duration  # total audio length in seconds
        segments = list(segments_gen)    # materialise the lazy generator

        timestamps = [
            {
                "start": round(seg.start, 3),
                "end":   round(seg.end,   3),
                "text":  seg.text.strip(),
            }
            for seg in segments
        ]

        full_text = " ".join(t["text"] for t in timestamps)

        APP_LOGGER.info(
            f"Transcription complete for '{filename}': "
            f"{len(segments)} segments, {duration:.1f}s total"
        )

        return {
            "text": full_text,
            "metadata": {
                "source":     filename,
                "duration":   round(duration, 3),
                "timestamps": timestamps,
            },
        }


# Module-level singleton (model loaded lazily on first use)
audio_transcriber = AudioTranscriber()
