from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from uniface import RetinaFace, ArcFace, compute_similarity

app = FastAPI(title="Face Verification API")

# CORS — allow MarketHub frontend + backend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8002",
        "http://localhost:3000",  # MarketHub backend
        "http://localhost:5174",  # MarketHub frontend dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()

class CompareFacesRequest(BaseModel):
    image1_url: str  # Avatar URL
    image2_url: str  # Selfie URL

class CompareFacesResponse(BaseModel):
    verified: bool
    similarity: float
    threshold: float
    message: str

class VerifyIdentityRequest(BaseModel):
    avatar_url: str   # Registered avatar/profile image URL
    selfie_url: str   # Live selfie image URL

class VerifyIdentityResponse(BaseModel):
    verified: bool
    similarity: float
    threshold: float
    message: str
    confidence_level: str  # "high", "medium", "low"

def download_image(url: str) -> np.ndarray:
    """Download image from URL and convert to OpenCV format"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array and BGR for OpenCV
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Face Verification API", "status": "running", "services": ["e-motel", "market-hub"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/compare-faces", response_model=CompareFacesResponse)
async def compare_faces(request: CompareFacesRequest):
    """
    Compare two face images and return similarity score (E-Motel legacy endpoint)
    """
    try:
        # Download images
        image1 = download_image(request.image1_url)
        image2 = download_image(request.image2_url)
        
        # Detect faces
        faces1 = detector.detect(image1)
        faces2 = detector.detect(image2)
        
        # Check if faces detected
        if not faces1 or len(faces1) == 0:
            raise HTTPException(status_code=400, detail="No face detected in avatar image")
        
        if not faces2 or len(faces2) == 0:
            raise HTTPException(status_code=400, detail="No face detected in selfie image")
        
        # Get embeddings
        embedding1 = recognizer.get_normalized_embedding(image1, faces1[0]['landmarks'])
        embedding2 = recognizer.get_normalized_embedding(image2, faces2[0]['landmarks'])
        
        # Compute similarity
        similarity = compute_similarity(embedding1, embedding2)
        
        # Threshold for verification (adjust as needed)
        threshold = 0.7
        verified = similarity >= threshold
        
        message = "Faces match - Identity verified" if verified else "Faces do not match"
        
        return CompareFacesResponse(
            verified=verified,
            similarity=float(similarity),
            threshold=threshold,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")


@app.post("/api/verify-identity", response_model=VerifyIdentityResponse)
async def verify_identity(request: VerifyIdentityRequest):
    """
    Verify user identity for MarketHub blue-tick verification.
    Compares a registered avatar/profile photo with a live selfie.
    Returns verification result with confidence level.
    """
    try:
        # Download images
        avatar_img = download_image(request.avatar_url)
        selfie_img = download_image(request.selfie_url)
        
        # Detect faces
        avatar_faces = detector.detect(avatar_img)
        selfie_faces = detector.detect(selfie_img)
        
        # Validate face detection
        if not avatar_faces or len(avatar_faces) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in profile/avatar image. Please use a clear face photo."
            )
        
        if not selfie_faces or len(selfie_faces) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in selfie. Please ensure your face is clearly visible."
            )
        
        # Get embeddings from the most prominent face
        avatar_embedding = recognizer.get_normalized_embedding(avatar_img, avatar_faces[0]['landmarks'])
        selfie_embedding = recognizer.get_normalized_embedding(selfie_img, selfie_faces[0]['landmarks'])
        
        # Compute similarity
        similarity = compute_similarity(avatar_embedding, selfie_embedding)
        sim_float = float(similarity)
        
        # Threshold for MarketHub verification (stricter than general comparison)
        threshold = 0.72
        verified = sim_float >= threshold
        
        # Confidence levels
        if sim_float >= 0.85:
            confidence_level = "high"
        elif sim_float >= 0.72:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        if verified:
            message = f"Identity verified successfully (confidence: {confidence_level})"
        else:
            message = "Identity verification failed. The photos do not match sufficiently."
        
        return VerifyIdentityResponse(
            verified=verified,
            similarity=sim_float,
            threshold=threshold,
            message=message,
            confidence_level=confidence_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity verification failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
