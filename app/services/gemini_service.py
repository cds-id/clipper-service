import json
import re
from google import genai
from google.genai import types
from app.models.schemas import TranscriptSegment, KeyPoint
from app.config import get_settings


class GeminiService:
    def __init__(self):
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Lazy load the Gemini client."""
        if self._client is None:
            settings = get_settings()
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    def _format_transcript_with_timestamps(
        self,
        segments: list[TranscriptSegment]
    ) -> str:
        """Format transcript with timestamps for Gemini analysis."""
        lines = []
        for seg in segments:
            timestamp = f"[{seg.start:.1f}s - {seg.end:.1f}s]"
            lines.append(f"{timestamp} {seg.text}")
        return "\n".join(lines)

    async def extract_key_points(
        self,
        segments: list[TranscriptSegment],
        max_clips: int = 5,
        min_duration: float = 10.0,
        max_duration: float = 120.0
    ) -> list[KeyPoint]:
        """
        Analyze transcript and extract key points with timestamp ranges.
        """
        client = self._get_client()
        transcript = self._format_transcript_with_timestamps(segments)

        prompt = f"""Analyze the following video transcript and identify the {max_clips} most important key points or moments.

For each key point, provide:
1. A short title (max 10 words)
2. A brief summary (1-2 sentences)
3. The start timestamp (in seconds)
4. The end timestamp (in seconds)
5. An importance score (1-10)

Requirements:
- Each clip should be between {min_duration} and {max_duration} seconds
- Focus on the most insightful, educational, or engaging moments
- Ensure clips don't overlap
- Order by importance (most important first)

Return your response as a JSON array with this exact format:
[
  {{
    "title": "Key Point Title",
    "summary": "Brief description of this key point",
    "start_time": 0.0,
    "end_time": 30.0,
    "importance": 8
  }}
]

TRANSCRIPT:
{transcript}

Return ONLY the JSON array, no other text."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=4096,
            )
        )
        response_text = response.text.strip()

        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            key_points_data = json.loads(response_text)
            return [KeyPoint(**kp) for kp in key_points_data]
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                key_points_data = json.loads(json_match.group())
                return [KeyPoint(**kp) for kp in key_points_data]
            raise ValueError("Failed to parse Gemini response as JSON")


gemini_service = GeminiService()
