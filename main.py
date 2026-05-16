# main.py
# This is the complete FastAPI application.
# It connects data_loader, search_engine, and chatbot
# into one working API with proper endpoints.
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from search_engine import initialize_search_system
from chatbot import chat

# -------------------------------------------------------
# STEP A: Define request and response models
# -------------------------------------------------------
# These classes define exactly what JSON our API
# accepts and returns. Pydantic validates everything
# automatically — if the data is wrong format,
# it returns a helpful error message.

class ChatRequest(BaseModel):
    """
    What the user sends TO our API.
    
    Example request body:
    {
        "message": "I need to hire a software engineer",
        "conversation_id": "abc123",
        "conversation_history": []
    }
    """
    message: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[dict]] = []


class AssessmentRecommendation(BaseModel):
    """
    One SHL assessment recommendation.
    This is the exact format the assignment requires.
    """
    name: str
    description: str
    job_levels: str
    duration_minutes: int
    test_type: str
    remote_testing: str
    adaptive: str


class ChatResponse(BaseModel):
    """
    What our API sends BACK to the user.
    
    Example response:
    {
        "response": "Based on your needs...",
        "recommendations": [...],
        "conversation_id": "abc123",
        "message_count": 1
    }
    """
    response: str
    recommendations: List[AssessmentRecommendation]
    conversation_id: str
    message_count: int


# -------------------------------------------------------
# STEP B: Global variables for our AI system
# -------------------------------------------------------
# We load the model and index ONCE when server starts.
# This saves time — loading takes a few seconds,
# so we don't want to reload on every request!

ml_model = None      # sentence-transformer model
faiss_index = None   # FAISS search index
assessments_df = None  # our CSV data as a dataframe


# -------------------------------------------------------
# STEP C: Lifespan — runs on startup and shutdown
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs when the server STARTS.
    We use it to load our AI model and FAISS index
    so they are ready before any requests come in.
    """
    global ml_model, faiss_index, assessments_df
    
    print("🚀 Starting SHL Assessment Recommender...")
    print("📊 Loading AI model and search index...")
    
    # Initialize everything
    ml_model, faiss_index, assessments_df = initialize_search_system()
    
    print("✅ Server ready to accept requests!")
    
    yield  # Server runs here
    
    # Cleanup when server stops (optional)
    print("👋 Server shutting down...")


# -------------------------------------------------------
# STEP D: Create FastAPI app
# -------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="AI-powered chatbot that recommends SHL assessments for recruiters",
    version="1.0.0",
    lifespan=lifespan
)

# Allow requests from any origin (needed for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# STEP E: Endpoints
# -------------------------------------------------------

@app.get("/")
def home():
    """Root endpoint — confirms API is running"""
    return {
        "message": "SHL Assessment Recommender API is running!",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "assessments": "GET /assessments",
            "docs": "GET /docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns whether the AI model is loaded and ready.
    """
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "index_loaded": faiss_index is not None,
        "assessments_loaded": assessments_df is not None,
        "total_assessments": len(assessments_df) if assessments_df is not None else 0
    }


@app.get("/assessments")
def get_all_assessments():
    """
    Returns all SHL assessments in our database.
    Useful for browsing available assessments.
    """
    if assessments_df is None:
        raise HTTPException(
            status_code=503,
            detail="Assessment data not loaded yet"
        )
    
    assessments = []
    for _, row in assessments_df.iterrows():
        assessments.append({
            "name": row["name"],
            "description": row["description"],
            "job_levels": row["job_levels"],
            "duration_minutes": int(row["duration_minutes"]),
            "test_type": row["test_type"],
            "remote_testing": row["remote_testing"],
            "adaptive": row["adaptive"]
        })
    
    return {
        "total": len(assessments),
        "assessments": assessments
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint — the heart of our application.
    
    Accepts a recruiter message and returns:
    - AI response with recommendations
    - List of recommended assessments
    - Conversation ID for tracking
    - Message count
    
    Example request:
    {
        "message": "I need to hire a software engineer",
        "conversation_id": null,
        "conversation_history": []
    }
    """
    # Check if AI system is loaded
    if ml_model is None or faiss_index is None:
        raise HTTPException(
            status_code=503,
            detail="AI system not ready yet. Please wait a moment."
        )
    
    # Generate conversation ID if not provided
    # This helps track multi-turn conversations
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Get conversation history (previous messages)
    history = request.conversation_history or []
    
    # Count messages so far
    message_count = len(history) // 2 + 1
    
    try:
        # Call our chatbot function
        result = chat(
            user_message=request.message,
            conversation_history=history,
            model=ml_model,
            index=faiss_index,
            df=assessments_df
        )
        
        # Format recommendations to match our schema
        formatted_recommendations = []
        for rec in result["recommendations"]:
            formatted_recommendations.append(
                AssessmentRecommendation(
                    name=rec["name"],
                    description=rec["description"],
                    job_levels=rec["job_levels"],
                    duration_minutes=rec["duration_minutes"],
                    test_type=rec["test_type"],
                    remote_testing=rec["remote_testing"],
                    adaptive=rec["adaptive"]
                )
            )
        
        # Return the response
        return ChatResponse(
            response=result["response"],
            recommendations=formatted_recommendations,
            conversation_id=conversation_id,
            message_count=message_count
        )
    
    except Exception as e:
        # If something goes wrong, return a helpful error
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )