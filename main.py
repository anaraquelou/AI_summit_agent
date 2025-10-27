from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is required")
    print("Please run: python setup.py")
    sys.exit(1)

app = FastAPI(title="Return Policy Chat Agent", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    message: str
    conversation_history: List[ChatMessage]
    status: str = "success"

# Initialize the agent (will be imported from agent module)
from agent.return_agent import ReturnPolicyAgent

agent = ReturnPolicyAgent()

@app.get("/")
async def root():
    return {"message": "Return Policy Chat Agent API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Process the message through the agent
        response = agent.process_message(
            message=request.message,
            conversation_history=request.conversation_history
        )
        
        return ChatResponse(
            message=response["message"],
            conversation_history=response["conversation_history"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
