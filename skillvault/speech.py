"""Optional local speech transcription; recordings are never sent to an API."""

from functools import lru_cache
from io import BytesIO


class TranscriptionError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_speech_model(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise TranscriptionError("Install the optional speech dependency: pip install -r requirements-speech.txt") from error
    try:
        return WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)
    except Exception as error:
        raise TranscriptionError("The local speech model is not installed. Complete the speech setup in the README first.") from error


def transcribe_recording(data: bytes, model_name: str = "base", model=None) -> str:
    if not data or len(data) > 50 * 1024 * 1024:
        raise TranscriptionError("Upload a nonempty recording smaller than 50 MB.")
    try:
        recognizer = model if model is not None else load_speech_model(model_name)
        segments, info = recognizer.transcribe(BytesIO(data), beam_size=5, vad_filter=True)
        lines = []
        for segment in segments:
            if segment.text.strip():
                lines.append(f"[{segment.start:.1f}s–{segment.end:.1f}s] {segment.text.strip()}")
        if not lines:
            raise TranscriptionError("No speech was detected. Try a clearer recording or type your explanation.")
        return "\n".join(lines)
    except TranscriptionError:
        raise
    except Exception as error:
        raise TranscriptionError(f"Could not transcribe this recording: {error}") from error
