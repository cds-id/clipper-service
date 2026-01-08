# Clipper Service

A FastAPI-based video processing service that extracts key moments from videos using AI-powered transcription and summarization.

## Features

- **Audio Extraction**: Separates audio from video using FFmpeg
- **Speech-to-Text**: Transcribes audio with word-level timestamps using Whisper
- **AI Analysis**: Identifies key points and generates summaries using Google Gemini
- **Smart Trimming**: Automatically creates clips based on key moments
- **Auto Captions**: Burns word-level captions into video clips
- **Background Processing**: Async job queue with status polling

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Transcription**: faster-whisper (local Whisper)
- **AI/Summarization**: Google Gemini 1.5 Flash
- **Video Processing**: FFmpeg
- **Output Format**: MP4 (H.264)

## Requirements

- Python 3.10+
- FFmpeg 6.x+
- Google Gemini API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd clipper-service

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Configuration

Edit the `.env` file with your settings:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Whisper model (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL_SIZE=base

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Processing limits
MAX_VIDEO_SIZE_MB=500
ALLOWED_EXTENSIONS=mp4,mov,avi,mkv,webm
```

## Usage

### Start the Server

```bash
# Development
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Process a Video

```bash
# Upload and start processing
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@video.mp4" \
  -F "max_clips=5" \
  -F "include_captions=true"

# Response
{
  "job_id": "abc123-...",
  "status": "pending",
  "progress": 0
}
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"

# Response
{
  "job_id": "abc123-...",
  "status": "completed",
  "progress": 100,
  "result": {
    "key_points": [...],
    "clips": [...]
  }
}
```

### Download Clips

```bash
# Download captioned clip
curl -o clip1.mp4 "http://localhost:8000/api/v1/jobs/{job_id}/clips/0?captioned=true"

# Download without captions
curl -o clip1.mp4 "http://localhost:8000/api/v1/jobs/{job_id}/clips/0?captioned=false"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/process` | Upload video and start processing |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and results |
| `GET` | `/api/v1/jobs/{job_id}/clips/{index}` | Download a specific clip |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete job and cleanup files |

## Processing Pipeline

```
1. Upload Video
       │
       ▼
2. Extract Audio (FFmpeg → WAV 16kHz mono)
       │
       ▼
3. Transcribe (Whisper → word-level timestamps)
       │
       ▼
4. Analyze (Gemini → key points + time ranges)
       │
       ▼
5. Trim Video (FFmpeg → clips per key point)
       │
       ▼
6. Add Captions (FFmpeg + ASS subtitles)
       │
       ▼
7. Return Results
```

## Project Structure

```
clipper-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings management
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── ffmpeg_service.py    # Video/audio processing
│   │   ├── transcription.py     # Whisper transcription
│   │   ├── gemini_service.py    # AI analysis
│   │   └── processor.py         # Pipeline orchestrator
│   └── utils/
│       └── helpers.py       # Utility functions
├── storage/
│   ├── uploads/             # Uploaded videos
│   ├── temp/                # Intermediate files
│   └── output/              # Processed clips
├── jobs/                    # Job status files
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Request Parameters

### POST /api/v1/process

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Video file to process |
| `max_clips` | int | 5 | Maximum number of clips to extract (1-20) |
| `min_clip_duration` | float | 10.0 | Minimum clip duration in seconds |
| `max_clip_duration` | float | 120.0 | Maximum clip duration in seconds |
| `include_captions` | bool | true | Add captions to clips |

## Job Status Values

| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `processing` | Initial processing |
| `extracting_audio` | Extracting audio from video |
| `transcribing` | Transcribing audio to text |
| `analyzing` | AI analyzing transcript |
| `trimming` | Creating video clips |
| `adding_captions` | Burning captions into clips |
| `completed` | Processing finished |
| `failed` | Processing failed |

## License

MIT
