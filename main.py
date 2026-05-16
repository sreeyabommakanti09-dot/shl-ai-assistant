# main.py
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from search_engine import initialize_search_system, search_assessments
from chatbot import generate_response

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------
# Request and Response Models
# -------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# -------------------------------------------------------
# Global variables
# -------------------------------------------------------
vectorizer = None
tfidf_matrix = None
assessments_df = None

# -------------------------------------------------------
# Lifespan
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorizer, tfidf_matrix, assessments_df
    print("🚀 Starting SHL Assessment Recommender...")
    vectorizer, tfidf_matrix, assessments_df = initialize_search_system()
    print("✅ Server ready to accept requests!")
    yield
    print("👋 Server shutting down...")

# -------------------------------------------------------
# FastAPI app
# -------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="AI-powered SHL assessment recommendation chatbot",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "SHL Assessment Recommender API is running!",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="AI system not ready yet."
        )

    # Convert messages to history format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Get the last user message
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    last_user_message = user_messages[-1]["content"]

    # Build conversation history (all except last message)
    history = messages[:-1]

    try:
        # Search for relevant assessments
        search_results = search_assessments(
            query=last_user_message,
            model=vectorizer,
            index=tfidf_matrix,
            df=assessments_df,
            top_k=5
        )

        # Generate AI response
        ai_reply = generate_response(
            user_message=last_user_message,
            conversation_history=history,
            search_results=search_results
        )

        # Format recommendations
        recommendations = []
        # Only include recommendations if AI has enough context
        vague_words = ["what", "which", "tell me", "more", "clarify", "?"]
        is_clarifying = any(word in ai_reply.lower() for word in vague_words) and len(ai_reply) < 300

        if not is_clarifying:
            for result in search_results[:5]:
                recommendations.append(
                    Recommendation(
                        name=result["name"],
                        url=result["url"],
                        test_type=result["test_type"]
                    )
                )

        # Detect end of conversation
        end_phrases = ["good luck", "best of luck", "happy hiring", "all the best"]
        end_of_conversation = any(phrase in ai_reply.lower() for phrase in end_phrases)

        return ChatResponse(
            reply=ai_reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )