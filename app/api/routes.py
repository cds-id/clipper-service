import shutil
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from app.models.schemas import (
    JobResponse, JobStatus, ProcessRequest, HealthResponse,
    ProcessUrlRequest, VideoInfoResponse
)
from app.services.processor import video_processor
from app.services.download_service import download_service
from app.utils.helpers import (
    generate_job_id, get_file_extension, save_job_status, load_job_status
)
from app.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Video Processing"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        whisper_model=settings.whisper_model_size
    )


@router.post("/process", response_model=JobResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_clips: int = Form(default=5),
    min_clip_duration: float = Form(default=10.0),
    max_clip_duration: float = Form(default=120.0),
    include_captions: bool = Form(default=True),
    caption_style: str = Form(default="default"),
    caption_mode: str = Form(default="clipper")
):
    """
    Upload a video file and start processing.
    Returns a job ID for tracking progress.

    Caption styles: default, neon, fire, ocean, minimal
    Caption modes: clipper (word-by-word highlight), karaoke (smooth fill)
    """
    settings = get_settings()

    # Validate file extension
    extension = get_file_extension(file.filename or "")
    if extension not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
        )

    # Generate job ID and save file
    job_id = generate_job_id()
    upload_path = settings.upload_path / f"{job_id}.{extension}"

    # Save uploaded file
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check file size
    file_size = upload_path.stat().st_size
    if file_size > settings.max_video_size_bytes:
        upload_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_video_size_mb}MB"
        )

    # Create initial job record
    now = datetime.utcnow()
    job = JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        message="Job created, waiting to start...",
        created_at=now,
        updated_at=now
    )
    await save_job_status(settings.jobs_path, job_id, job)

    # Create process request
    request = ProcessRequest(
        max_clips=max_clips,
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
        include_captions=include_captions,
        caption_style=caption_style,
        caption_mode=caption_mode
    )

    # Start background processing
    background_tasks.add_task(
        process_video_file_task,
        job_id,
        upload_path,
        request
    )

    return job


def process_video_file_task(job_id: str, video_path: Path, request: ProcessRequest):
    """Background task to process uploaded video file."""
    import asyncio

    async def _run():
        try:
            await video_processor.process_video(job_id, video_path, request)
        except Exception as e:
            settings = get_settings()
            job = await load_job_status(settings.jobs_path, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.message = f"Processing failed: {str(e)}"
                job.updated_at = datetime.utcnow()
                await save_job_status(settings.jobs_path, job_id, job)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


@router.post("/process-url", response_model=JobResponse)
async def process_video_from_url(
    background_tasks: BackgroundTasks,
    request: ProcessUrlRequest
):
    """
    Process video from URL (YouTube, YouTube Shorts, or direct video link).
    Supports:
    - YouTube: youtube.com/watch?v=..., youtu.be/...
    - YouTube Shorts: youtube.com/shorts/...
    - Direct links: https://example.com/video.mp4
    - Other sites supported by yt-dlp (Twitter, TikTok, etc.)
    """
    settings = get_settings()
    job_id = generate_job_id()

    # Create initial job record
    now = datetime.utcnow()
    job = JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        message="Job created, preparing to download...",
        created_at=now,
        updated_at=now
    )
    await save_job_status(settings.jobs_path, job_id, job)

    # Create process request
    process_request = ProcessRequest(
        max_clips=request.max_clips,
        min_clip_duration=request.min_clip_duration,
        max_clip_duration=request.max_clip_duration,
        include_captions=request.include_captions,
        caption_style=request.caption_style,
        caption_mode=request.caption_mode
    )

    # Start background processing with download
    background_tasks.add_task(
        process_video_from_url_task,
        job_id,
        request.url,
        process_request
    )

    return job


def process_video_from_url_task(
    job_id: str,
    url: str,
    request: ProcessRequest
):
    """Background task to download and process video from URL."""
    import asyncio

    async def _run():
        settings = get_settings()
        try:
            # Update status to downloading
            from app.utils.helpers import update_job_progress
            await update_job_progress(
                settings.jobs_path, job_id, JobStatus.DOWNLOADING, 5,
                "Downloading video..."
            )

            # Download video
            video_path, source_type = await download_service.download(
                url,
                settings.upload_path,
                job_id
            )

            await update_job_progress(
                settings.jobs_path, job_id, JobStatus.PROCESSING, 10,
                f"Download complete ({source_type}). Starting processing..."
            )

            # Process the downloaded video
            await video_processor.process_video(job_id, video_path, request)

        except Exception as e:
            # Mark job as failed
            job = await load_job_status(settings.jobs_path, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.message = f"Processing failed: {str(e)}"
                job.updated_at = datetime.utcnow()
                await save_job_status(settings.jobs_path, job_id, job)

    # Run the async function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


@router.get("/video-info", response_model=VideoInfoResponse)
async def get_video_info(url: str):
    """
    Get video information without downloading.
    Useful for previewing video details before processing.
    """
    try:
        info = await download_service.get_video_info(url)
        return VideoInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to get video info: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    settings = get_settings()
    job = await load_job_status(settings.jobs_path, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.get("/jobs/{job_id}/clips/{clip_index}")
async def download_clip(job_id: str, clip_index: int, captioned: bool = True):
    """Download a specific clip from a completed job."""
    settings = get_settings()
    job = await load_job_status(settings.jobs_path, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )

    if not job.result or clip_index >= len(job.result.clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip = job.result.clips[clip_index]
    clip_path = Path(clip.captioned_clip_path if captioned else clip.clip_path)

    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file not found")

    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        filename=clip_path.name
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    settings = get_settings()
    job = await load_job_status(settings.jobs_path, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete associated files
    # Upload file
    for ext in settings.allowed_extensions_list + ['mp4', 'mkv', 'webm']:
        upload_file = settings.upload_path / f"{job_id}.{ext}"
        if upload_file.exists():
            upload_file.unlink()

    # Output clips
    if job.result:
        for clip in job.result.clips:
            clip_path = Path(clip.clip_path)
            captioned_path = Path(clip.captioned_clip_path)
            if clip_path.exists():
                clip_path.unlink()
            if captioned_path.exists() and captioned_path != clip_path:
                captioned_path.unlink()

    # Job file
    job_file = settings.jobs_path / f"{job_id}.json"
    if job_file.exists():
        job_file.unlink()

    return {"message": "Job deleted successfully"}
