# E-Motel Face Verification API

Face verification service using Uniface for identity verification.

## Setup

1. Install uv:
```bash
pip install uv
```

2. Create virtual environment and install dependencies:
```bash
uv venv
.venv\Scripts\activate  # Windows
uv pip install -e .
```

## Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

### POST /api/compare-faces

Compare two face images.

**Request:**
```json
{
  "image1_url": "https://cloudinary.com/avatar.jpg",
  "image2_url": "https://cloudinary.com/selfie.jpg"
}
```

**Response:**
```json
{
  "verified": true,
  "similarity": 0.95,
  "threshold": 0.7,
  "message": "Faces match - Identity verified"
}
```

## Environment

- Python 3.10+
- FastAPI
- Uniface
- OpenCV
