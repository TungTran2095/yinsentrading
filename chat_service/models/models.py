"""
Database models for the Chat AI service
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message model for chat interactions"""
    id: Optional[str] = Field(None, description="Message ID")
    user_id: str = Field(..., description="User ID")
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role (user or assistant)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    intent: Optional[str] = Field(None, description="Detected intent")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Extracted entities")
    confidence: Optional[float] = Field(None, description="Intent confidence score")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "content": "What is the current price of Bitcoin?",
                "role": "user",
                "intent": "get_price",
                "entities": [{"entity": "crypto", "value": "Bitcoin"}],
                "confidence": 0.92
            }
        }


class Conversation(BaseModel):
    """Conversation model for storing chat history"""
    id: Optional[str] = Field(None, description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    title: Optional[str] = Field(None, description="Conversation title")
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "title": "Bitcoin price inquiry",
                "messages": [
                    {
                        "user_id": "user123",
                        "content": "What is the current price of Bitcoin?",
                        "role": "user",
                        "intent": "get_price",
                        "entities": [{"entity": "crypto", "value": "Bitcoin"}],
                        "confidence": 0.92
                    },
                    {
                        "user_id": "user123",
                        "content": "The current price of Bitcoin is $60,123.45",
                        "role": "assistant"
                    }
                ]
            }
        }


class Intent(BaseModel):
    """Intent model for intent classification"""
    name: str = Field(..., description="Intent name")
    description: str = Field(..., description="Intent description")
    examples: List[str] = Field(..., description="Example phrases for this intent")
    required_entities: Optional[List[str]] = Field(None, description="Required entities for this intent")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "get_price",
                "description": "Get the current price of a cryptocurrency or stock",
                "examples": [
                    "What is the price of Bitcoin?",
                    "Show me the current value of ETH",
                    "How much is Tesla stock worth right now?"
                ],
                "required_entities": ["asset"]
            }
        }


class Entity(BaseModel):
    """Entity model for entity extraction"""
    name: str = Field(..., description="Entity name")
    description: str = Field(..., description="Entity description")
    examples: Dict[str, List[str]] = Field(..., description="Example values for this entity")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "asset",
                "description": "Cryptocurrency or stock symbol/name",
                "examples": {
                    "Bitcoin": ["BTC", "Bitcoin", "XBT"],
                    "Ethereum": ["ETH", "Ethereum", "Ether"],
                    "Tesla": ["TSLA", "Tesla"]
                }
            }
        }


class ChatRequest(BaseModel):
    """Chat request model for API"""
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuing a conversation")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "message": "What is the current price of Bitcoin?",
                "conversation_id": None
            }
        }


class ChatResponse(BaseModel):
    """Chat response model for API"""
    conversation_id: str = Field(..., description="Conversation ID")
    message: str = Field(..., description="Assistant response")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Intent confidence score")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data related to the response")
    
    class Config:
        schema_extra = {
            "example": {
                "conversation_id": "conv123",
                "message": "The current price of Bitcoin is $60,123.45",
                "intent": "get_price",
                "confidence": 0.92,
                "data": {
                    "price": 60123.45,
                    "currency": "USD",
                    "timestamp": "2023-05-01T12:34:56Z"
                }
            }
        }
