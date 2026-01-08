from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    TRIMMING = "trimming"
    ADDING_CAPTIONS = "adding_captions"
    COMPLETED = "completed"
    FAILED = "failed"


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = []


class TranscriptionInfo(BaseModel):
    language: str
    language_probability: float
    duration: float


class KeyPoint(BaseModel):
    title: str
    summary: str
    start_time: float
    end_time: float
    importance: int = Field(ge=1, le=10, default=5)


class VideoMetadata(BaseModel):
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    size_bytes: int


class ClipResult(BaseModel):
    key_point: KeyPoint
    clip_path: str
    captioned_clip_path: str


class ProcessingResult(BaseModel):
    job_id: str
    original_video: VideoMetadata
    transcription_info: TranscriptionInfo
    transcript: list[TranscriptSegment]
    key_points: list[KeyPoint]
    clips: list[ClipResult]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100, default=0)
    message: str = ""
    created_at: datetime
    updated_at: datetime
    result: ProcessingResult | None = None
    error: str | None = None


class ProcessRequest(BaseModel):
    max_clips: int = Field(default=5, ge=1, le=20)
    min_clip_duration: float = Field(default=10.0, ge=5.0)
    max_clip_duration: float = Field(default=120.0, le=300.0)
    include_captions: bool = True
    caption_style: str = Field(
        default="default",
        description="Caption color style: default, neon, fire, ocean, minimal"
    )
    caption_mode: str = Field(
        default="clipper",
        description="Caption mode: clipper (word highlight) or karaoke (smooth fill)"
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    whisper_model: str


class ProcessUrlRequest(BaseModel):
    url: str = Field(..., description="Video URL (YouTube, YouTube Shorts, or direct link)")
    max_clips: int = Field(default=5, ge=1, le=20)
    min_clip_duration: float = Field(default=10.0, ge=5.0)
    max_clip_duration: float = Field(default=120.0, le=300.0)
    include_captions: bool = True
    caption_style: str = Field(
        default="default",
        description="Caption color style: default, neon, fire, ocean, minimal"
    )
    caption_mode: str = Field(
        default="clipper",
        description="Caption mode: clipper (word highlight) or karaoke (smooth fill)"
    )


class VideoInfoResponse(BaseModel):
    title: str
    duration: float
    uploader: str
    thumbnail: str | None = None
    description: str = ""
