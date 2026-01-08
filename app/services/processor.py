from pathlib import Path
from datetime import datetime
from app.models.schemas import (
    JobStatus, JobResponse, ProcessRequest, ProcessingResult,
    ClipResult, TranscriptSegment
)
from app.services.ffmpeg_service import ffmpeg_service
from app.services.transcription import transcription_service
from app.services.gemini_service import gemini_service
from app.utils.helpers import (
    save_job_status, update_job_progress, sanitize_filename
)
from app.config import get_settings


class VideoProcessor:
    def __init__(self):
        self.settings = get_settings()

    async def process_video(
        self,
        job_id: str,
        video_path: Path,
        request: ProcessRequest
    ) -> None:
        """
        Main processing pipeline that orchestrates all services.
        """
        jobs_path = self.settings.jobs_path
        temp_path = self.settings.temp_path
        output_path = self.settings.output_path

        try:
            # Step 1: Extract video metadata
            await update_job_progress(
                jobs_path, job_id, JobStatus.PROCESSING, 5,
                "Extracting video metadata..."
            )
            video_metadata = await ffmpeg_service.get_video_metadata(video_path)

            # Check video duration limit
            if video_metadata.duration > self.settings.max_video_duration_sec:
                max_min = self.settings.max_video_duration_min
                actual_min = video_metadata.duration / 60
                raise ValueError(
                    f"Video too long: {actual_min:.1f} min. Maximum allowed: {max_min} min"
                )

            # Step 2: Extract audio
            await update_job_progress(
                jobs_path, job_id, JobStatus.EXTRACTING_AUDIO, 15,
                "Extracting audio from video..."
            )
            audio_path = await ffmpeg_service.extract_audio(video_path, temp_path)

            # Step 3: Transcribe audio (auto-detect language)
            await update_job_progress(
                jobs_path, job_id, JobStatus.TRANSCRIBING, 30,
                "Transcribing audio (this may take a while)..."
            )
            transcript, transcription_info = await transcription_service.transcribe(audio_path)

            # Update with detected language
            await update_job_progress(
                jobs_path, job_id, JobStatus.ANALYZING, 50,
                f"Detected language: {transcription_info.language} ({transcription_info.language_probability:.1%}). Analyzing transcript..."
            )
            key_points = await gemini_service.extract_key_points(
                transcript,
                max_clips=request.max_clips,
                min_duration=request.min_clip_duration,
                max_duration=request.max_clip_duration
            )

            # Step 5: Create clips
            clips: list[ClipResult] = []
            total_clips = len(key_points)

            for i, key_point in enumerate(key_points):
                progress = 60 + int((i / total_clips) * 30)

                # Trim video
                await update_job_progress(
                    jobs_path, job_id, JobStatus.TRIMMING, progress,
                    f"Creating clip {i + 1}/{total_clips}..."
                )

                clip_name = sanitize_filename(f"{job_id}_clip_{i + 1}_{key_point.title[:20]}")
                clip_path = await ffmpeg_service.trim_video(
                    video_path,
                    output_path,
                    key_point.start_time,
                    key_point.end_time,
                    clip_name
                )

                captioned_clip_path = clip_path  # Default to same path

                # Add captions if requested
                if request.include_captions:
                    await update_job_progress(
                        jobs_path, job_id, JobStatus.ADDING_CAPTIONS, progress + 5,
                        f"Adding captions to clip {i + 1}/{total_clips}..."
                    )

                    # Get words for this time range
                    words = transcription_service.get_words_in_range(
                        transcript,
                        key_point.start_time,
                        key_point.end_time
                    )

                    if words:
                        captioned_clip_path = await ffmpeg_service.add_captions(
                            clip_path,
                            output_path,
                            words,
                            video_metadata,
                            clip_name,
                            offset=key_point.start_time,
                            style=request.caption_style,
                            caption_mode=request.caption_mode
                        )

                clips.append(ClipResult(
                    key_point=key_point,
                    clip_path=str(clip_path),
                    captioned_clip_path=str(captioned_clip_path)
                ))

            # Step 6: Create result
            result = ProcessingResult(
                job_id=job_id,
                original_video=video_metadata,
                transcription_info=transcription_info,
                transcript=transcript,
                key_points=key_points,
                clips=clips
            )

            # Update job as completed
            job = await self._get_job(jobs_path, job_id)
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.message = "Processing complete!"
            job.result = result
            job.updated_at = datetime.utcnow()
            await save_job_status(jobs_path, job_id, job)

            # Cleanup temp files
            if audio_path.exists():
                audio_path.unlink()

        except Exception as e:
            # Update job as failed
            await self._mark_job_failed(jobs_path, job_id, str(e))
            raise

    async def _get_job(self, jobs_path: Path, job_id: str) -> JobResponse:
        from app.utils.helpers import load_job_status
        job = await load_job_status(jobs_path, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        return job

    async def _mark_job_failed(
        self,
        jobs_path: Path,
        job_id: str,
        error: str
    ) -> None:
        job = await self._get_job(jobs_path, job_id)
        job.status = JobStatus.FAILED
        job.error = error
        job.message = "Processing failed"
        job.updated_at = datetime.utcnow()
        await save_job_status(jobs_path, job_id, job)


video_processor = VideoProcessor()
