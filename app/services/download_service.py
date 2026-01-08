import re
import asyncio
from pathlib import Path
from urllib.parse import urlparse
import yt_dlp
import aiofiles
import httpx
from app.config import get_settings


class DownloadService:
    def __init__(self):
        self.settings = get_settings()

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube link (including Shorts)."""
        youtube_patterns = [
            r'(youtube\.com/watch\?v=)',
            r'(youtube\.com/shorts/)',
            r'(youtu\.be/)',
            r'(youtube\.com/embed/)',
            r'(youtube\.com/v/)',
        ]
        return any(re.search(pattern, url) for pattern in youtube_patterns)

    def _is_direct_video_url(self, url: str) -> bool:
        """Check if URL is a direct video file link."""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in video_extensions)

    def _get_url_type(self, url: str) -> str:
        """Determine the type of URL."""
        if self._is_youtube_url(url):
            if '/shorts/' in url:
                return 'youtube_shorts'
            return 'youtube'
        elif self._is_direct_video_url(url):
            return 'direct'
        else:
            # Try yt-dlp for other supported sites
            return 'other'

    async def download_from_youtube(
        self,
        url: str,
        output_path: Path,
        job_id: str
    ) -> Path:
        """Download video from YouTube or YouTube Shorts."""
        output_file = output_path / f"{job_id}.mp4"

        ydl_opts = {
            # Limit to 480p max to reduce file size and processing time
            'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best',
            'outtmpl': str(output_file),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        def _download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # Run in thread pool to not block async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download)

        # yt-dlp might add extension, find the actual file
        if output_file.exists():
            return output_file

        # Check for file with different naming
        for file in output_path.glob(f"{job_id}.*"):
            if file.suffix.lower() in ['.mp4', '.mkv', '.webm']:
                return file

        raise FileNotFoundError(f"Downloaded file not found for job {job_id}")

    async def download_direct_url(
        self,
        url: str,
        output_path: Path,
        job_id: str
    ) -> Path:
        """Download video from direct URL."""
        parsed = urlparse(url)
        extension = Path(parsed.path).suffix or '.mp4'
        output_file = output_path / f"{job_id}{extension}"

        async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
            async with client.stream('GET', url) as response:
                response.raise_for_status()
                async with aiofiles.open(output_file, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)

        return output_file

    async def download_with_ytdlp(
        self,
        url: str,
        output_path: Path,
        job_id: str
    ) -> Path:
        """Download video using yt-dlp (supports many sites)."""
        output_template = str(output_path / f"{job_id}.%(ext)s")

        ydl_opts = {
            # Limit to 480p max to reduce file size and processing time
            'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best',
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }

        def _download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download)

        # Find downloaded file
        for file in output_path.glob(f"{job_id}.*"):
            if file.suffix.lower() in ['.mp4', '.mkv', '.webm', '.mov']:
                return file

        raise FileNotFoundError(f"Downloaded file not found for job {job_id}")

    async def download(
        self,
        url: str,
        output_path: Path,
        job_id: str
    ) -> tuple[Path, str]:
        """
        Download video from URL (auto-detect source type).
        Returns tuple of (file_path, source_type).
        """
        url_type = self._get_url_type(url)

        if url_type in ['youtube', 'youtube_shorts']:
            file_path = await self.download_from_youtube(url, output_path, job_id)
        elif url_type == 'direct':
            file_path = await self.download_direct_url(url, output_path, job_id)
        else:
            # Try yt-dlp for other sites (Twitter, TikTok, etc.)
            file_path = await self.download_with_ytdlp(url, output_path, job_id)

        return file_path, url_type

    async def get_video_info(self, url: str) -> dict:
        """Get video information without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        def _extract():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _extract)

        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'thumbnail': info.get('thumbnail'),
            'description': info.get('description', ''),
        }


download_service = DownloadService()
