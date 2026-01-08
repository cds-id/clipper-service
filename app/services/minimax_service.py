import requests
from pathlib import Path
from app.config import get_settings


class MinimaxService:
    def __init__(self):
        self.api_url = "https://api.minimax.io/v1/music_generation"
        self._api_key = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            settings = get_settings()
            self._api_key = settings.minimax_api_key
        return self._api_key

    async def generate_instrumental(
        self,
        prompt: str,
        output_path: Path,
        clip_name: str
    ) -> Path:
        """
        Generate instrumental background music using MiniMax Music 2.0.

        Args:
            prompt: Music style description (genre, mood, tempo, instruments)
            output_path: Directory to save the generated music
            clip_name: Name for the output file

        Returns:
            Path to the generated MP3 file
        """
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("MINIMAX_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Use instrumental tags for instrumental-only generation (min 10 chars required)
        payload = {
            "model": "music-2.0",
            "prompt": prompt,
            "lyrics": "[inst]\n[instrumental]\n[outro]",
            "audio_setting": {
                "sample_rate": 44100,
                "bitrate": 256000,
                "format": "mp3"
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=300  # 5 minute timeout for music generation
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = response.text if response else str(e)
            raise ValueError(f"MiniMax API error: {response.status_code} - {error_detail}")

        try:
            data = response.json()
        except Exception:
            raise ValueError(f"Invalid JSON response from MiniMax: {response.text[:500]}")

        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid response format from MiniMax: {data}")

        if "data" not in data or not data.get("data") or "audio" not in data.get("data", {}):
            # Check for error message in response
            error_msg = data.get("base_resp", {}).get("status_msg", str(data))
            raise ValueError(f"MiniMax API error: {error_msg}")

        audio_hex = data["data"]["audio"]

        music_file = output_path / f"{clip_name}_music.mp3"
        with open(music_file, "wb") as f:
            f.write(bytes.fromhex(audio_hex))

        return music_file


minimax_service = MinimaxService()
