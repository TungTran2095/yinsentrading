"""
Intent recognition service for natural language understanding
"""
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from spacy.tokens import Doc

import config
from models import Intent, Entity

logger = logging.getLogger(__name__)

class IntentRecognitionService:
    """Service for intent recognition and entity extraction"""
    
    def __init__(self):
        """Initialize NLP models and load intents"""
        # Load embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
            logger.info(f"Loaded spaCy model: {config.SPACY_MODEL}")
        except OSError:
            logger.info(f"Downloading spaCy model: {config.SPACY_MODEL}")
            spacy.cli.download(config.SPACY_MODEL)
            self.nlp = spacy.load(config.SPACY_MODEL)
        
        # Load intents and entities
        self.intents = self._load_intents()
        self.entities = self._load_entities()
        
        # Precompute embeddings for intent examples
        self.intent_embeddings = {}
        for intent_name, intent in self.intents.items():
            examples = intent.examples
            if examples:
                embeddings = self.embedding_model.encode(examples)
                self.intent_embeddings[intent_name] = embeddings
        
        logger.info(f"Loaded {len(self.intents)} intents and {len(self.entities)} entity types")
    
    def _load_intents(self) -> Dict[str, Intent]:
        """Load intent definitions from file"""
        intents_file = os.path.join(os.path.dirname(__file__), "../data/intents.json")
        try:
            with open(intents_file, "r") as f:
                intents_data = json.load(f)
            
            intents = {}
            for intent_data in intents_data:
                intent = Intent(**intent_data)
                intents[intent.name] = intent
            
            return intents
        except FileNotFoundError:
            logger.warning(f"Intents file not found: {intents_file}")
            # Create default intents
            return self._create_default_intents()
    
    def _create_default_intents(self) -> Dict[str, Intent]:
        """Create default intents if file not found"""
        intents = {
            "get_price": Intent(
                name="get_price",
                description="Get the current price of a cryptocurrency or stock",
                examples=[
                    "What is the price of Bitcoin?",
                    "Show me the current value of ETH",
                    "How much is Tesla stock worth right now?",
                    "Tell me the price of BTC",
                    "What's the current rate for Ethereum?",
                    "Check the price of Bitcoin for me"
                ],
                required_entities=["asset"]
            ),
            "get_portfolio": Intent(
                name="get_portfolio",
                description="Get information about the user's portfolio",
                examples=[
                    "Show me my portfolio",
                    "What's in my portfolio?",
                    "How is my portfolio performing?",
                    "What assets do I own?",
                    "Show portfolio summary",
                    "What's my current balance?"
                ],
                required_entities=[]
            ),
            "create_bot": Intent(
                name="create_bot",
                description="Create a new trading bot",
                examples=[
                    "Create a new trading bot",
                    "I want to make a new bot",
                    "Set up a bot for Bitcoin trading",
                    "Create a bot that trades ETH",
                    "Make me a new trading algorithm",
                    "I need a new automated trader"
                ],
                required_entities=["asset"]
            ),
            "bot_status": Intent(
                name="bot_status",
                description="Get the status of a trading bot",
                examples=[
                    "How are my bots doing?",
                    "What's the status of my Bitcoin bot?",
                    "Is my ETH bot running?",
                    "Show me the performance of my bots",
                    "Are any of my bots making money?",
                    "Check the status of my trading bots"
                ],
                required_entities=[]
            ),
            "market_analysis": Intent(
                name="market_analysis",
                description="Get market analysis for a cryptocurrency or stock",
                examples=[
                    "What's your analysis of Bitcoin?",
                    "Analyze the Ethereum market",
                    "Give me a market overview for BTC",
                    "What do you think about Tesla stock?",
                    "Should I buy Bitcoin now?",
                    "Provide market insights for ETH"
                ],
                required_entities=["asset"]
            ),
            "help": Intent(
                name="help",
                description="Get help with using the system",
                examples=[
                    "Help me",
                    "I need help",
                    "What can you do?",
                    "Show me available commands",
                    "How do I use this?",
                    "What are your capabilities?"
                ],
                required_entities=[]
            ),
            "greeting": Intent(
                name="greeting",
                description="Greet the user",
                examples=[
                    "Hello",
                    "Hi there",
                    "Hey",
                    "Good morning",
                    "Good afternoon",
                    "Good evening"
                ],
                required_entities=[]
            )
        }
        
        # Save default intents to file
        intents_file = os.path.join(os.path.dirname(__file__), "../data/intents.json")
        os.makedirs(os.path.dirname(intents_file), exist_ok=True)
        with open(intents_file, "w") as f:
            json.dump([intent.dict() for intent in intents.values()], f, indent=2)
        
        return intents
    
    def _load_entities(self) -> Dict[str, Entity]:
        """Load entity definitions from file"""
        entities_file = os.path.join(os.path.dirname(__file__), "../data/entities.json")
        try:
            with open(entities_file, "r") as f:
                entities_data = json.load(f)
            
            entities = {}
            for entity_data in entities_data:
                entity = Entity(**entity_data)
                entities[entity.name] = entity
            
            return entities
        except FileNotFoundError:
            logger.warning(f"Entities file not found: {entities_file}")
            # Create default entities
            return self._create_default_entities()
    
    def _create_default_entities(self) -> Dict[str, Entity]:
        """Create default entities if file not found"""
        entities = {
            "asset": Entity(
                name="asset",
                description="Cryptocurrency or stock symbol/name",
                examples={
                    "Bitcoin": ["BTC", "Bitcoin", "XBT", "bitcoin"],
                    "Ethereum": ["ETH", "Ethereum", "Ether", "ethereum"],
                    "Litecoin": ["LTC", "Litecoin", "litecoin"],
                    "Ripple": ["XRP", "Ripple", "ripple"],
                    "Cardano": ["ADA", "Cardano", "cardano"],
                    "Tesla": ["TSLA", "Tesla", "tesla"],
                    "Apple": ["AAPL", "Apple", "apple"],
                    "Amazon": ["AMZN", "Amazon", "amazon"],
                    "Google": ["GOOGL", "Google", "google"],
                    "Microsoft": ["MSFT", "Microsoft", "microsoft"]
                }
            ),
            "timeframe": Entity(
                name="timeframe",
                description="Time period for analysis or trading",
                examples={
                    "1m": ["1 minute", "1min", "1 min", "one minute"],
                    "5m": ["5 minutes", "5min", "5 mins", "five minutes"],
                    "15m": ["15 minutes", "15min", "15 mins", "fifteen minutes"],
                    "30m": ["30 minutes", "30min", "30 mins", "thirty minutes"],
                    "1h": ["1 hour", "1hr", "one hour", "hourly"],
                    "4h": ["4 hours", "4hr", "4hrs", "four hours"],
                    "1d": ["1 day", "daily", "one day", "day"],
                    "1w": ["1 week", "weekly", "one week", "week"],
                    "1M": ["1 month", "monthly", "one month", "month"]
                }
            ),
            "strategy": Entity(
                name="strategy",
                description="Trading strategy type",
                examples={
                    "ensemble": ["ensemble learning", "ensemble", "machine learning", "ML"],
                    "rl": ["reinforcement learning", "RL", "reinforcement", "deep learning"],
                    "combined": ["combined", "hybrid", "ensemble and RL", "ML and RL"]
                }
            )
        }
        
        # Save default entities to file
        entities_file = os.path.join(os.path.dirname(__file__), "../data/entities.json")
        os.makedirs(os.path.dirname(entities_file), exist_ok=True)
        with open(entities_file, "w") as f:
            json.dump([entity.dict() for entity in entities.values()], f, indent=2)
        
        return entities
    
    def recognize_intent(self, text: str) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
        """
        Recognize intent and extract entities from text
        
        Returns:
            Tuple of (intent_name, confidence, entities)
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Get text embedding
        text_embedding = self.embedding_model.encode([text])[0]
        
        # Find closest intent
        best_intent = None
        best_score = 0.0
        
        for intent_name, embeddings in self.intent_embeddings.items():
            # Calculate cosine similarity with each example
            similarities = np.dot(embeddings, text_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            
            # Get max similarity
            max_similarity = np.max(similarities)
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent_name
        
        # Check if confidence is above threshold
        if best_score < config.INTENT_THRESHOLD:
            logger.info(f"No intent recognized with confidence above threshold. Best: {best_intent} ({best_score:.2f})")
            return None, best_score, entities
        
        logger.info(f"Recognized intent: {best_intent} with confidence {best_score:.2f}")
        return best_intent, best_score, entities
    
    def _extract_entities(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract entities from spaCy document"""
        extracted_entities = []
        
        # Extract named entities from spaCy
        for ent in doc.ents:
            extracted_entities.append({
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })
        
        # Extract custom entities using pattern matching
        for entity_type, entity in self.entities.items():
            for canonical, variations in entity.examples.items():
                for variation in variations:
                    if variation.lower() in doc.text.lower():
                        # Simple case-insensitive substring match
                        # In a production system, you would use more sophisticated matching
                        extracted_entities.append({
                            "entity": entity_type,
                            "value": canonical,
                            "text": variation,
                            "source": "custom"
                        })
        
        return extracted_entities
