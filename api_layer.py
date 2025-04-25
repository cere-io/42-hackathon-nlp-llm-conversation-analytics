from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
import uvicorn
import json
from datetime import datetime

# Import agent classes (we'll adapt these imports later)
try:
    from claude35 import Claude35ConversationDetector, Message as ClaudeMessage
    claude_available = True
except ImportError:
    claude_available = False

try:
    from gpt4 import GPT4ConversationDetector, Message as GPTMessage
    gpt4_available = True
except ImportError:
    gpt4_available = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conversation Analysis API", 
              description="API for analyzing conversations using different LLM agents")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request/response
class Author(BaseModel):
    id: str = Field(..., example="530192978", description="User ID")
    username: str = Field(..., example="ninja_by", description="Username")
    first_name: Optional[str] = Field("", example="Sergey", description="First name")
    last_name: Optional[str] = Field("", example="", description="Last name")
    is_bot: Optional[bool] = Field(False, example=False, description="Whether the user is a bot")

    class Config:
        schema_extra = {
            "example": {
                "id": "530192978",
                "username": "ninja_by",
                "first_name": "Sergey",
                "last_name": "",
                "is_bot": False
            }
        }

class Message(BaseModel):
    message_id: str = Field(..., example="MessageId(longValue=9)", description="Unique message identifier")
    message_text: str = Field(..., example="Is this working on stage?", description="Message content")
    message_timestamp: str = Field(..., example="1745577699", description="Message timestamp")
    author: Author = Field(..., description="Message author information")
    group_id: Optional[str] = Field(None, example="-4673616689", description="Group identifier")
    group_title: Optional[str] = Field(None, example="NLP Bot Test", description="Group title")

    # For backward compatibility with old formats
    @property
    def id(self) -> str:
        return self.message_id
    
    @property
    def text(self) -> str:
        return self.message_text
    
    @property
    def timestamp(self) -> str:
        return self.message_timestamp
    
    @property
    def username(self) -> str:
        return self.author.username
    
    class Config:
        schema_extra = {
            "example": {
                "message_id": "MessageId(longValue=9)",
                "group_id": "-4673616689",
                "group_title": "NLP Bot Test",
                "message_text": "Is this working on stage?",
                "message_timestamp": "1745577699",
                "author": {
                    "id": "530192978",
                    "username": "ninja_by",
                    "first_name": "Sergey",
                    "last_name": "",
                    "is_bot": False
                }
            }
        }

class AnalysisRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to analyze")
    model: str = Field("gpt4", example="gpt4", description="Model to use for analysis (gpt4 or claude35)")
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "message_id": "MessageId(longValue=9)",
                        "group_id": "-4673616689", 
                        "group_title": "NLP Bot Test",
                        "message_text": "Is this working on stage?",
                        "message_timestamp": "1745577699",
                        "author": {
                            "id": "530192978",
                            "username": "ninja_by",
                            "first_name": "Sergey",
                            "last_name": "",
                            "is_bot": False
                        }
                    },
                    {
                        "message_id": "MessageId(longValue=10)",
                        "group_id": "-4673616689",
                        "group_title": "NLP Bot Test",
                        "message_text": "Yes, it seems to be working fine!",
                        "message_timestamp": "1745577750",
                        "author": {
                            "id": "987654321",
                            "username": "user2",
                            "first_name": "Jane",
                            "last_name": "Smith",
                            "is_bot": False
                        }
                    }
                ],
                "model": "gpt4"
            }
        }

class Label(BaseModel):
    message_id: str = Field(..., example="MessageId(longValue=9)", description="Message ID")
    conversation_id: str = Field(..., example="conv123", description="Conversation ID")
    topic: str = Field(..., example="Stage Testing", description="Conversation topic")
    timestamp: str = Field(..., example="1745577699", description="Timestamp")
    labeler_id: str = Field(..., example="gpt4", description="ID of the labeler model")
    confidence: float = Field(..., example=0.95, description="Confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "message_id": "MessageId(longValue=9)",
                "conversation_id": "conv123",
                "topic": "Stage Testing",
                "timestamp": "1745577699",
                "labeler_id": "gpt4",
                "confidence": 0.95
            }
        }

class AnalysisResponse(BaseModel):
    labels: List[Label] = Field(..., description="Analysis results")
    model: str = Field(..., example="gpt4", description="Model used for analysis")
    processing_time: float = Field(..., example=1.23, description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "labels": [
                    {
                        "message_id": "MessageId(longValue=9)",
                        "conversation_id": "conv123",
                        "topic": "Stage Testing",
                        "timestamp": "1745577699",
                        "labeler_id": "gpt4",
                        "confidence": 0.95
                    },
                    {
                        "message_id": "MessageId(longValue=10)",
                        "conversation_id": "conv123",
                        "topic": "Stage Testing",
                        "timestamp": "1745577750",
                        "labeler_id": "gpt4",
                        "confidence": 0.95
                    }
                ],
                "model": "gpt4",
                "processing_time": 1.23
            }
        }

@app.get("/")
async def root():
    return {"status": "active", "available_models": get_available_models()}

@app.get("/health")
async def health():
    return {"status": "healthy", "available_models": get_available_models()}

def get_available_models():
    """Return list of available models based on imports and API keys"""
    models = []
    if gpt4_available and os.getenv("OPENAI_API_KEY"):
        models.append("gpt4")
    if claude_available and os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude35")
    return models

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_conversations(request: AnalysisRequest):
    """Analyze a conversation using specified model"""
    start_time = datetime.now()
    
    if request.model == "claude35":
        if not claude_available:
            raise HTTPException(status_code=400, detail="Claude 3.5 model not available")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
        
        detector = Claude35ConversationDetector()
        message_class = ClaudeMessage
        
    elif request.model == "gpt4":
        if not gpt4_available:
            raise HTTPException(status_code=400, detail="GPT-4 model not available")
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        
        detector = GPT4ConversationDetector()
        message_class = GPTMessage
        
    else:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not supported")
    
    # Convert DTO to internal model
    messages = []
    for msg in request.messages:
        # Create user dict compatible with the detector's expected format
        user = {
            "username": msg.author.username,
            "first_name": msg.author.first_name,
            "last_name": msg.author.last_name
        }
        # Create message using the model-specific message class
        messages.append(message_class(
            msg.message_id, 
            msg.message_text, 
            msg.message_timestamp, 
            user
        ))
    
    # Perform detection
    try:
        labels = detector.detect(messages)
        
        # Convert back to DTO
        label_dtos = []
        for label in labels:
            label_dtos.append(Label(
                message_id=label.message_id,
                conversation_id=label.conversation_id,
                topic=label.topic,
                timestamp=label.timestamp,
                labeler_id=label.metadata.get('labeler_id', request.model),
                confidence=label.metadata.get('confidence', 1.0)
            ))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return AnalysisResponse(
            labels=label_dtos,
            model=request.model,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api_layer:app", host="0.0.0.0", port=port, reload=False) 