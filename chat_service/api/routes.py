"""
API routes for the Chat AI service
"""
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from bson.objectid import ObjectId

from models import ChatRequest, ChatResponse, Conversation, Message
from services import MongoDBService, IntentRecognitionService, ResponseService
import config

router = APIRouter(prefix=config.API_PREFIX)
logger = logging.getLogger(__name__)

# Initialize services
db_service = MongoDBService()
intent_service = IntentRecognitionService()
response_service = ResponseService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and generate a response
    """
    try:
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = db_service.create_conversation(request.user_id)
        else:
            # Verify conversation exists
            conversation = db_service.get_conversation(conversation_id)
            if not conversation:
                conversation_id = db_service.create_conversation(request.user_id)
        
        # Recognize intent and extract entities
        intent, confidence, entities = intent_service.recognize_intent(request.message)
        
        # Store user message
        user_message = {
            "user_id": request.user_id,
            "content": request.message,
            "role": "user",
            "intent": intent,
            "entities": entities,
            "confidence": confidence
        }
        db_service.add_message(conversation_id, user_message)
        
        # Generate response
        response_data = response_service.generate_response(
            intent, entities, request.user_id, request.message
        )
        
        # Store assistant message
        assistant_message = {
            "user_id": request.user_id,
            "content": response_data["message"],
            "role": "assistant"
        }
        db_service.add_message(conversation_id, assistant_message)
        
        # Update conversation title if it's a new conversation
        conversation = db_service.get_conversation(conversation_id)
        if not conversation.get("title") and intent:
            # Use the first user message as the title
            title = request.message[:50] + ("..." if len(request.message) > 50 else "")
            db_service.update_conversation_title(conversation_id, title)
        
        # Return response
        return ChatResponse(
            conversation_id=conversation_id,
            message=response_data["message"],
            intent=intent,
            confidence=confidence,
            data=response_data.get("data")
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

@router.get("/conversations/{user_id}", response_model=List[Dict[str, Any]])
async def get_conversations(user_id: str, limit: int = 10):
    """
    Get conversations for a user
    """
    try:
        conversations = db_service.get_conversations_by_user(user_id, limit)
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversations: {str(e)}"
        )

@router.get("/conversations/{user_id}/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(user_id: str, conversation_id: str):
    """
    Get a specific conversation
    """
    try:
        conversation = db_service.get_conversation(conversation_id)
        if not conversation or conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return conversation
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation: {str(e)}"
        )

@router.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    """
    Delete a conversation
    """
    try:
        conversation = db_service.get_conversation(conversation_id)
        if not conversation or conversation.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        success = db_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete conversation"
            )
        
        return JSONResponse(content={"message": "Conversation deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}"
        )
