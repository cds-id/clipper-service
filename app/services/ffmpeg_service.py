import subprocess
import json
from pathlib import Path
from app.models.schemas import VideoMetadata, WordTimestamp
from app.utils.helpers import format_timestamp, format_ass_timestamp


# Clipper-style color presets
CAPTION_STYLES = {
    "default": {
        "primary": "&H00FFFFFF",      # White
        "highlight": "&H0000FFFF",    # Yellow
        "outline": "&H00000000",      # Black
        "shadow": "&H80000000",       # Semi-transparent black
    },
    "neon": {
        "primary": "&H00FFFFFF",      # White
        "highlight": "&H0000FF00",    # Green
        "outline": "&H00FF00FF",      # Magenta
        "shadow": "&H80000000",
    },
    "fire": {
        "primary": "&H00FFFFFF",      # White
        "highlight": "&H000080FF",    # Orange
        "outline": "&H000000FF",      # Red
        "shadow": "&H80000000",
    },
    "ocean": {
        "primary": "&H00FFFFFF",      # White
        "highlight": "&H00FFFF00",    # Cyan
        "outline": "&H00FF8800",      # Blue
        "shadow": "&H80000000",
    },
    "minimal": {
        "primary": "&H00FFFFFF",      # White
        "highlight": "&H00FFFFFF",    # White (no highlight change)
        "outline": "&H00000000",      # Black
        "shadow": "&H00000000",
    },
}


class FFmpegService:
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        self.ffprobe_path = "ffprobe"

    async def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from video file using ffprobe."""
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"),
            None
        )

        if not video_stream:
            raise ValueError("No video stream found in file")

        fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

        return VideoMetadata(
            filename=video_path.name,
            duration=float(data["format"]["duration"]),
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            fps=fps,
            size_bytes=int(data["format"]["size"])
        )

    async def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Extract audio from video as WAV file for transcription."""
        audio_output = output_path / f"{video_path.stem}_audio.wav"

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(audio_output)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return audio_output

    async def trim_video(
        self,
        video_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        clip_name: str
    ) -> Path:
        """Trim video to specified time range."""
        output_file = output_path / f"{clip_name}.mp4"

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-ss", format_timestamp(start_time),
            "-to", format_timestamp(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-y",
            str(output_file)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_file

    def _split_into_phrases(
        self,
        words: list[WordTimestamp],
        max_words: int = 6,
        max_gap: float = 1.0
    ) -> list[list[WordTimestamp]]:
        """
        Split words into natural phrases based on timing gaps and length.
        """
        if not words:
            return []

        phrases = []
        current_phrase = [words[0]]

        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]

            # Check for natural break (gap > max_gap seconds or phrase too long)
            gap = curr_word.start - prev_word.end
            if gap > max_gap or len(current_phrase) >= max_words:
                phrases.append(current_phrase)
                current_phrase = [curr_word]
            else:
                current_phrase.append(curr_word)

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def _generate_clipper_style_ass(
        self,
        words: list[WordTimestamp],
        video_width: int,
        video_height: int,
        offset: float = 0.0,
        style: str = "default",
        words_per_line: int = 6
    ) -> str:
        """
        Generate clipper-style ASS subtitles with word-by-word highlighting.

        Features:
        - Shows full phrase while highlighting current word
        - Large, bold centered text
        - Word-by-word highlight animation (current word in different color/size)
        - Previous words remain visible
        - Modern styling with outlines and shadows
        """
        colors = CAPTION_STYLES.get(style, CAPTION_STYLES["default"])

        # Calculate font size based on video resolution (responsive)
        base_font_size = max(int(video_height * 0.07), 42)  # 7% of height, min 42

        # ASS header with clipper-style formatting
        ass_content = f"""[Script Info]
Title: Clipper Style Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,{base_font_size},{colors['primary']},{colors['highlight']},{colors['outline']},{colors['shadow']},-1,0,0,0,100,100,0,0,1,4,2,2,20,20,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        if not words:
            return ass_content

        # Split into natural phrases
        phrases = self._split_into_phrases(words, max_words=words_per_line)

        # Generate dialogue lines - show full phrase with current word highlighted
        for phrase in phrases:
            # For each word in the phrase, create a frame showing full phrase
            for word_idx, current_word in enumerate(phrase):
                word_start = max(0, current_word.start - offset)
                word_end = current_word.end - offset

                # Build the full phrase with current word highlighted
                line_parts = []
                for idx, w in enumerate(phrase):
                    if idx == word_idx:
                        # Current word - highlighted with color and scale
                        line_parts.append(
                            f"{{\\c{colors['highlight']}\\fscx115\\fscy115}}"
                            f"{w.word.upper()}"
                            f"{{\\c{colors['primary']}\\fscx100\\fscy100}}"
                        )
                    elif idx < word_idx:
                        # Previous words - shown normally (already spoken)
                        line_parts.append(w.word.upper())
                    else:
                        # Future words - slightly dimmed
                        line_parts.append(
                            f"{{\\alpha&H40&}}{w.word.upper()}{{\\alpha&H00&}}"
                        )

                text = " ".join(line_parts)

                ass_content += (
                    f"Dialogue: 0,"
                    f"{format_ass_timestamp(word_start)},"
                    f"{format_ass_timestamp(word_end)},"
                    f"Default,,0,0,0,,"
                    f"{text}\n"
                )

        return ass_content

    def _generate_karaoke_ass(
        self,
        words: list[WordTimestamp],
        video_width: int,
        video_height: int,
        offset: float = 0.0,
        style: str = "default"
    ) -> str:
        """
        Generate karaoke-style ASS with smooth word fill animation.
        """
        colors = CAPTION_STYLES.get(style, CAPTION_STYLES["default"])
        base_font_size = max(int(video_height * 0.08), 48)

        ass_content = f"""[Script Info]
Title: Karaoke Style Captions
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,{base_font_size},{colors['primary']},{colors['highlight']},{colors['outline']},{colors['shadow']},-1,0,0,0,100,100,0,0,1,4,2,2,20,20,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        if not words:
            return ass_content

        # Group into lines of 3-4 words
        words_per_line = 3
        for i in range(0, len(words), words_per_line):
            chunk = words[i:i + words_per_line]
            if not chunk:
                continue

            chunk_start = max(0, chunk[0].start - offset)
            chunk_end = chunk[-1].end - offset

            # Build karaoke text with \kf tags (smooth fill)
            karaoke_parts = []
            for w in chunk:
                # Duration in centiseconds
                duration_cs = int((w.end - w.start) * 100)
                karaoke_parts.append(f"{{\\kf{duration_cs}}}{w.word.upper()}")

            text = " ".join(karaoke_parts)

            ass_content += (
                f"Dialogue: 0,"
                f"{format_ass_timestamp(chunk_start)},"
                f"{format_ass_timestamp(chunk_end)},"
                f"Karaoke,,0,0,0,,"
                f"{text}\n"
            )

        return ass_content

    async def add_captions(
        self,
        video_path: Path,
        output_path: Path,
        words: list[WordTimestamp],
        video_metadata: VideoMetadata,
        clip_name: str,
        offset: float = 0.0,
        style: str = "default",
        caption_mode: str = "clipper"  # "clipper" or "karaoke"
    ) -> Path:
        """
        Burn clipper-style captions into video with word-level timing.

        Args:
            video_path: Input video file
            output_path: Directory for output
            words: List of words with timestamps
            video_metadata: Video metadata
            clip_name: Name for output files
            offset: Time offset for word timestamps
            style: Caption style preset ("default", "neon", "fire", "ocean", "minimal")
            caption_mode: "clipper" for word-by-word highlight, "karaoke" for smooth fill
        """
        # Generate ASS subtitle file based on mode
        if caption_mode == "karaoke":
            ass_content = self._generate_karaoke_ass(
                words,
                video_metadata.width,
                video_metadata.height,
                offset,
                style
            )
        else:
            ass_content = self._generate_clipper_style_ass(
                words,
                video_metadata.width,
                video_metadata.height,
                offset,
                style
            )

        ass_file = output_path / f"{clip_name}_subs.ass"
        with open(ass_file, "w", encoding="utf-8") as f:
            f.write(ass_content)

        output_file = output_path / f"{clip_name}_captioned.mp4"

        # Burn subtitles into video with high quality
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vf", f"ass='{str(ass_file)}'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",  # High quality
            "-c:a", "aac",
            "-b:a", "192k",
            "-y",
            str(output_file)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_file


ffmpeg_service = FFmpegService()
