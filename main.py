from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import sys
import sqlite3
from dotenv import load_dotenv

# Import the agent and related classes
from agent.return_agent import agent, AgentState, AgentContext
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader

# Get project root directory (main.py is at the root)
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "datasets" / "olist_ecommerce.db"
PDF_PATH = PROJECT_ROOT / "docs" / "BIX-return-policy.pdf"

# Load environment variables
load_dotenv()

app = FastAPI(title="Data Analyst Chat Agent", version="2.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"  # Thread ID for conversation memory

class ChatResponse(BaseModel):
    message: str
    status: str = "success"


@app.get("/")
async def root():
    return {
        "message": "Data analyst Chat Agent API",
        "version": "2.0.0",
        "description": "LangGraph-based agent with intelligent routing for data analysis"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conn = None
    try:
        # Prepare DB connection
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

        # Load PDF once per request
        loader = PyPDFLoader(str(PDF_PATH))
        docs = loader.load()
        pdf_text = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in docs
        )

        # Build runtime context
        runtime_context = AgentContext(
            user_id=request.thread_id,
            db_connection=conn,
            pdf_policy=pdf_text
        )

        # Add the new user message
        langchain_messages = [(HumanMessage(content=request.message))]

        # Create input state
        input_state: Dict[str, Any] = {
            "messages": langchain_messages,
            "pdf_context": "",
            "decide_path": "general",
        }
        
        # Create config with thread_id for checkpointing (conversation memory)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": request.thread_id
            }
        }
        
        # Invoke the agent
        # Using stream to get the final state with all messages
        final_state = None
        for state in agent.stream(
            input_state, 
            stream_mode="values", 
            config=config,
            context=runtime_context
            ):
            final_state = state
        
        if not final_state:
            raise HTTPException(status_code=500, detail="Agent returned no state")
        
        # Get the last message from the agent (should be an AIMessage)
        response_messages = final_state.get("messages", [])
        if not response_messages:
            raise HTTPException(status_code=500, detail="Agent returned no messages")
        
        # Find the last assistant message
        last_assistant_message = None
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage):
                last_assistant_message = msg
                break
        
        if not last_assistant_message:
            # If no AIMessage found, use the last message
            last_message = response_messages[-1]
            last_assistant_message = AIMessage(
                content=str(last_message.content) if hasattr(last_message, 'content') else "I'm processing your request."
            )
        
        # Extract content from the assistant message
        assistant_content = last_assistant_message.content
        if not isinstance(assistant_content, str):
            assistant_content = str(assistant_content)
        
        
        return ChatResponse(
            message=assistant_content,
            status="success"
        )
    except Exception as e:
        import traceback
        error_detail = str(e)
        print(f"Error in chat endpoint: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        if conn:
            conn.close()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_type": "LangGraph routing agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
