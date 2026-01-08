import uuid
import json
import aiofiles
from pathlib import Path
from datetime import datetime
from app.models.schemas import JobResponse, JobStatus


def generate_job_id() -> str:
    return str(uuid.uuid4())


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for FFmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_ass_timestamp(seconds: float) -> str:
    """Convert seconds to H:MM:SS.cc format for ASS subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


async def save_job_status(jobs_path: Path, job_id: str, job: JobResponse) -> None:
    job_file = jobs_path / f"{job_id}.json"
    async with aiofiles.open(job_file, "w") as f:
        await f.write(job.model_dump_json(indent=2))


async def load_job_status(jobs_path: Path, job_id: str) -> JobResponse | None:
    job_file = jobs_path / f"{job_id}.json"
    if not job_file.exists():
        return None
    async with aiofiles.open(job_file, "r") as f:
        content = await f.read()
        data = json.loads(content)
        return JobResponse(**data)


async def update_job_progress(
    jobs_path: Path,
    job_id: str,
    status: JobStatus,
    progress: int,
    message: str = ""
) -> None:
    job = await load_job_status(jobs_path, job_id)
    if job:
        job.status = status
        job.progress = progress
        job.message = message
        job.updated_at = datetime.utcnow()
        await save_job_status(jobs_path, job_id, job)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are unsafe for filenames."""
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, "_")
    return filename
