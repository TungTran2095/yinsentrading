"""
Database service for MongoDB operations
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo import MongoClient
from bson import ObjectId

from models import Message, Conversation
import config

logger = logging.getLogger(__name__)

class MongoDBService:
    """Service for MongoDB operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DB]
        self.collection = self.db[config.MONGODB_COLLECTION]
        logger.info(f"Connected to MongoDB: {config.MONGODB_URI}")
    
    def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation"""
        conversation = {
            "user_id": user_id,
            "title": title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = self.collection.insert_one(conversation)
        logger.info(f"Created new conversation with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        try:
            conversation = self.collection.find_one({"_id": ObjectId(conversation_id)})
            if conversation:
                conversation["id"] = str(conversation["_id"])
                del conversation["_id"]
            return conversation
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    def get_conversations_by_user(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations by user ID"""
        try:
            conversations = list(self.collection.find(
                {"user_id": user_id},
                sort=[("updated_at", -1)],
                limit=limit
            ))
            for conv in conversations:
                conv["id"] = str(conv["_id"])
                del conv["_id"]
            return conversations
        except Exception as e:
            logger.error(f"Error getting conversations for user {user_id}: {e}")
            return []
    
    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to a conversation"""
        try:
            message["timestamp"] = datetime.utcnow()
            result = self.collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": message},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            success = result.modified_count > 0
            if success:
                logger.info(f"Added message to conversation {conversation_id}")
            else:
                logger.warning(f"Failed to add message to conversation {conversation_id}")
            return success
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            return False
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the title of a conversation"""
        try:
            result = self.collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        "title": title,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated title of conversation {conversation_id}")
            else:
                logger.warning(f"Failed to update title of conversation {conversation_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating title of conversation {conversation_id}: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(conversation_id)})
            success = result.deleted_count > 0
            if success:
                logger.info(f"Deleted conversation {conversation_id}")
            else:
                logger.warning(f"Failed to delete conversation {conversation_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
