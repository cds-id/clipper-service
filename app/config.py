from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = ""
    hf_token: str = ""

    # Whisper Configuration
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Application Settings
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: str = "storage/uploads"
    temp_dir: str = "storage/temp"
    output_dir: str = "storage/output"
    jobs_dir: str = "jobs"

    # Processing Settings
    max_video_size_mb: int = 500
    allowed_extensions: str = "mp4,mov,avi,mkv,webm"

    @property
    def upload_path(self) -> Path:
        return self.base_dir / self.upload_dir

    @property
    def temp_path(self) -> Path:
        return self.base_dir / self.temp_dir

    @property
    def output_path(self) -> Path:
        return self.base_dir / self.output_dir

    @property
    def jobs_path(self) -> Path:
        return self.base_dir / self.jobs_dir

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
