import os
from pathlib import Path
from faster_whisper import WhisperModel
from app.models.schemas import TranscriptSegment, WordTimestamp, TranscriptionInfo
from app.config import get_settings


class TranscriptionService:
    def __init__(self):
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        """Lazy load the Whisper model."""
        if self._model is None:
            settings = get_settings()

            # Set HF_TOKEN for model downloads
            if settings.hf_token:
                os.environ["HF_TOKEN"] = settings.hf_token

            # Device and compute type from config
            self._model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type
            )
        return self._model

    async def transcribe(
        self,
        audio_path: Path
    ) -> tuple[list[TranscriptSegment], TranscriptionInfo]:
        """
        Transcribe audio file and return segments with word-level timestamps.
        Auto-detects language from audio.
        """
        model = self._get_model()

        # Auto-detect language (language=None)
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            language=None  # Auto-detect
        )

        # Get transcription info with detected language
        transcription_info = TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration
        )

        result = []
        for segment in segments:
            words = []
            if segment.words:
                for word in segment.words:
                    words.append(WordTimestamp(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end
                    ))

            result.append(TranscriptSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words
            ))

        return result, transcription_info

    def get_full_transcript(self, segments: list[TranscriptSegment]) -> str:
        """Combine all segments into a single transcript text."""
        return " ".join(seg.text for seg in segments)

    def get_words_in_range(
        self,
        segments: list[TranscriptSegment],
        start_time: float,
        end_time: float
    ) -> list[WordTimestamp]:
        """Get all words within a specific time range."""
        words = []
        for segment in segments:
            for word in segment.words:
                if start_time <= word.start <= end_time:
                    words.append(word)
        return words


transcription_service = TranscriptionService()
