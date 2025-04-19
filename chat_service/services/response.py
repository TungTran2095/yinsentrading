"""
Response generation service for chat interactions
"""
import logging
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

import config
from models import Intent, Entity

logger = logging.getLogger(__name__)

class ResponseService:
    """Service for generating responses to user queries"""
    
    def __init__(self):
        """Initialize response service"""
        logger.info("Initializing response service")
    
    def generate_response(self, intent: Optional[str], entities: List[Dict[str, Any]], 
                          user_id: str, message: str) -> Dict[str, Any]:
        """
        Generate a response based on intent and entities
        
        Args:
            intent: Recognized intent
            entities: Extracted entities
            user_id: User ID
            message: Original user message
            
        Returns:
            Dict containing response message and any additional data
        """
        if not intent:
            return self._handle_unknown_intent(message)
        
        # Handle different intents
        intent_handlers = {
            "greeting": self._handle_greeting,
            "help": self._handle_help,
            "get_price": self._handle_get_price,
            "get_portfolio": self._handle_get_portfolio,
            "create_bot": self._handle_create_bot,
            "bot_status": self._handle_bot_status,
            "market_analysis": self._handle_market_analysis
        }
        
        handler = intent_handlers.get(intent, self._handle_unknown_intent)
        return handler(entities, user_id, message)
    
    def _handle_unknown_intent(self, message: str) -> Dict[str, Any]:
        """Handle unknown intent"""
        responses = [
            "I'm not sure I understand what you're asking. Could you rephrase that?",
            "I don't have enough information to help with that. Can you provide more details?",
            "I'm still learning and don't quite understand that request. Can you try asking in a different way?",
            "I'm not sure how to respond to that. Try asking about cryptocurrency prices, your portfolio, or creating trading bots."
        ]
        
        import random
        return {
            "message": random.choice(responses),
            "data": None
        }
    
    def _handle_greeting(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle greeting intent"""
        greetings = [
            "Hello! How can I help you with your trading today?",
            "Hi there! I'm your trading assistant. What would you like to know?",
            "Greetings! I can help you with market information, portfolio management, and trading bots. What do you need?",
            "Welcome! I'm here to assist with your cryptocurrency and stock trading needs. How can I help?"
        ]
        
        import random
        return {
            "message": random.choice(greetings),
            "data": None
        }
    
    def _handle_help(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle help intent"""
        help_message = (
            "I can help you with the following:\n\n"
            "• Get cryptocurrency or stock prices (e.g., 'What's the price of Bitcoin?')\n"
            "• Check your portfolio (e.g., 'Show me my portfolio')\n"
            "• Create trading bots (e.g., 'Create a bot for Bitcoin trading')\n"
            "• Check bot status (e.g., 'How are my bots performing?')\n"
            "• Get market analysis (e.g., 'Analyze the Ethereum market')\n\n"
            "What would you like to do?"
        )
        
        return {
            "message": help_message,
            "data": {
                "capabilities": [
                    "price_check",
                    "portfolio_management",
                    "bot_creation",
                    "bot_monitoring",
                    "market_analysis"
                ]
            }
        }
    
    def _handle_get_price(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle get_price intent"""
        # Extract asset entity
        asset = None
        for entity in entities:
            if entity.get("entity") == "asset":
                asset = entity.get("value")
                break
        
        if not asset:
            return {
                "message": "Which cryptocurrency or stock would you like to check the price for?",
                "data": None
            }
        
        # Try to get price from data service
        try:
            response = requests.get(
                f"{config.DATA_SERVICE_URL}/market/price",
                params={"symbol": asset},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                price = data.get("price")
                currency = data.get("currency", "USD")
                timestamp = data.get("timestamp")
                
                # Format the price with commas for thousands
                formatted_price = f"${price:,.2f}" if price else "unavailable"
                
                return {
                    "message": f"The current price of {asset} is {formatted_price} {currency}.",
                    "data": data
                }
            else:
                # Fallback to mock data if service is unavailable
                logger.warning(f"Failed to get price from data service: {response.status_code}")
                return self._mock_price_data(asset)
                
        except requests.RequestException as e:
            logger.error(f"Error connecting to data service: {e}")
            return self._mock_price_data(asset)
    
    def _mock_price_data(self, asset: str) -> Dict[str, Any]:
        """Generate mock price data for demo purposes"""
        import random
        
        price_ranges = {
            "Bitcoin": (50000, 60000),
            "Ethereum": (3000, 4000),
            "Litecoin": (150, 250),
            "Ripple": (0.5, 1.5),
            "Cardano": (1, 3),
            "Tesla": (700, 900),
            "Apple": (150, 200),
            "Amazon": (3000, 3500),
            "Google": (2500, 3000),
            "Microsoft": (300, 350)
        }
        
        # Default range if asset not in dictionary
        price_range = price_ranges.get(asset, (100, 1000))
        price = round(random.uniform(*price_range), 2)
        
        return {
            "message": f"The current price of {asset} is ${price:,.2f} USD.",
            "data": {
                "price": price,
                "currency": "USD",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "mock"
            }
        }
    
    def _handle_get_portfolio(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle get_portfolio intent"""
        try:
            response = requests.get(
                f"{config.TRADING_SERVICE_URL}/users/{user_id}/portfolio",
                timeout=5
            )
            
            if response.status_code == 200:
                portfolio = response.json()
                
                # Format portfolio summary
                total_value = portfolio.get("total_value", 0)
                assets = portfolio.get("assets", [])
                
                if not assets:
                    return {
                        "message": "You don't have any assets in your portfolio yet. Would you like to create a trading bot to start building your portfolio?",
                        "data": portfolio
                    }
                
                # Create portfolio summary
                summary = f"Your portfolio is currently worth ${total_value:,.2f} USD.\n\n"
                summary += "Assets:\n"
                
                for asset in assets:
                    symbol = asset.get("symbol")
                    amount = asset.get("amount")
                    value = asset.get("value")
                    pnl_pct = asset.get("pnl_percentage")
                    
                    pnl_str = f"({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
                    summary += f"• {symbol}: {amount} (${value:,.2f} {pnl_str})\n"
                
                return {
                    "message": summary,
                    "data": portfolio
                }
            else:
                # Fallback to mock data
                logger.warning(f"Failed to get portfolio: {response.status_code}")
                return self._mock_portfolio_data(user_id)
                
        except requests.RequestException as e:
            logger.error(f"Error connecting to trading service: {e}")
            return self._mock_portfolio_data(user_id)
    
    def _mock_portfolio_data(self, user_id: str) -> Dict[str, Any]:
        """Generate mock portfolio data for demo purposes"""
        import random
        
        assets = [
            {
                "symbol": "BTC",
                "amount": round(random.uniform(0.1, 2.0), 4),
                "value": round(random.uniform(5000, 20000), 2),
                "pnl_percentage": round(random.uniform(-15, 25), 2)
            },
            {
                "symbol": "ETH",
                "amount": round(random.uniform(1.0, 10.0), 4),
                "value": round(random.uniform(3000, 10000), 2),
                "pnl_percentage": round(random.uniform(-10, 30), 2)
            },
            {
                "symbol": "TSLA",
                "amount": round(random.uniform(1.0, 20.0), 2),
                "value": round(random.uniform(1000, 5000), 2),
                "pnl_percentage": round(random.uniform(-20, 15), 2)
            }
        ]
        
        total_value = sum(asset["value"] for asset in assets)
        
        # Create portfolio summary
        summary = f"Your portfolio is currently worth ${total_value:,.2f} USD.\n\n"
        summary += "Assets:\n"
        
        for asset in assets:
            symbol = asset.get("symbol")
            amount = asset.get("amount")
            value = asset.get("value")
            pnl_pct = asset.get("pnl_percentage")
            
            pnl_str = f"({pnl_pct:+.2f}%)" if pnl_pct is not None else ""
            summary += f"• {symbol}: {amount} (${value:,.2f} {pnl_str})\n"
        
        return {
            "message": summary,
            "data": {
                "total_value": total_value,
                "assets": assets,
                "source": "mock"
            }
        }
    
    def _handle_create_bot(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle create_bot intent"""
        # Extract asset entity
        asset = None
        strategy = "combined"  # Default strategy
        
        for entity in entities:
            if entity.get("entity") == "asset":
                asset = entity.get("value")
            elif entity.get("entity") == "strategy":
                strategy = entity.get("value")
        
        if not asset:
            return {
                "message": "Which cryptocurrency or stock would you like to create a trading bot for?",
                "data": None
            }
        
        # Create response with link to bot creation page
        response_message = (
            f"I can help you create a trading bot for {asset} using our {strategy} strategy. "
            f"To get started, I've prepared a bot configuration for you.\n\n"
            f"You can customize and launch your bot from the bot creation page. "
            f"Would you like me to take you there now?"
        )
        
        return {
            "message": response_message,
            "data": {
                "action": "create_bot",
                "asset": asset,
                "strategy": strategy,
                "url": "/bots/create"
            }
        }
    
    def _handle_bot_status(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle bot_status intent"""
        # Extract asset entity (optional)
        asset = None
        for entity in entities:
            if entity.get("entity") == "asset":
                asset = entity.get("value")
        
        try:
            # Get all bots or filter by asset
            params = {"asset": asset} if asset else {}
            response = requests.get(
                f"{config.TRADING_SERVICE_URL}/users/{user_id}/bots",
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                bots = response.json()
                
                if not bots:
                    if asset:
                        return {
                            "message": f"You don't have any trading bots for {asset}. Would you like to create one?",
                            "data": {
                                "action": "suggest_create_bot",
                                "asset": asset
                            }
                        }
                    else:
                        return {
                            "message": "You don't have any trading bots yet. Would you like to create one?",
                            "data": {
                                "action": "suggest_create_bot"
                            }
                        }
                
                # Create bot status summary
                summary = f"You have {len(bots)} active trading bot{'s' if len(bots) != 1 else ''}:\n\n"
                
                for bot in bots:
                    name = bot.get("name")
                    status = bot.get("status")
                    symbol = bot.get("symbol")
                    pnl = bot.get("pnl", 0)
                    pnl_pct = bot.get("pnl_percentage", 0)
                    
                    status_emoji = "🟢" if status == "running" else "🔴" if status == "stopped" else "⚠️"
                    pnl_str = f"(${pnl:+,.2f}, {pnl_pct:+.2f}%)" if pnl is not None else ""
                    
                    summary += f"{status_emoji} {name}: Trading {symbol}, {status} {pnl_str}\n"
                
                return {
                    "message": summary,
                    "data": {
                        "bots": bots,
                        "action": "show_bots"
                    }
                }
            else:
                # Fallback to mock data
                logger.warning(f"Failed to get bots: {response.status_code}")
                return self._mock_bot_status(user_id, asset)
                
        except requests.RequestException as e:
            logger.error(f"Error connecting to trading service: {e}")
            return self._mock_bot_status(user_id, asset)
    
    def _mock_bot_status(self, user_id: str, asset: Optional[str] = None) -> Dict[str, Any]:
        """Generate mock bot status data for demo purposes"""
        import random
        
        # Generate random bots
        bot_templates = [
            {"name": "Bitcoin Trader", "symbol": "BTC/USDT", "asset": "Bitcoin"},
            {"name": "ETH Momentum", "symbol": "ETH/USDT", "asset": "Ethereum"},
            {"name": "LTC Swing Trader", "symbol": "LTC/USDT", "asset": "Litecoin"},
            {"name": "TSLA Day Trader", "symbol": "TSLA", "asset": "Tesla"}
        ]
        
        # Filter by asset if specified
        if asset:
            bot_templates = [bot for bot in bot_templates if bot["asset"] == asset]
            
            if not bot_templates:
                return {
                    "message": f"You don't have any trading bots for {asset}. Would you like to create one?",
                    "data": {
                        "action": "suggest_create_bot",
                        "asset": asset
                    }
                }
        
        # Generate 1-3 random bots
        num_bots = random.randint(1, min(3, len(bot_templates)))
        selected_templates = random.sample(bot_templates, num_bots)
        
        bots = []
        for template in selected_templates:
            status = random.choice(["running", "stopped"])
            pnl = round(random.uniform(-500, 1500), 2)
            initial_balance = 10000
            pnl_percentage = (pnl / initial_balance) * 100
            
            bot = {
                "id": f"bot_{random.randint(1000, 9999)}",
                "name": template["name"],
                "symbol": template["symbol"],
                "status": status,
                "strategy": random.choice(["ensemble", "rl", "combined"]),
                "pnl": pnl,
                "pnl_percentage": round(pnl_percentage, 2),
                "created_at": (datetime.utcnow().replace(
                    day=random.randint(1, 28),
                    hour=random.randint(0, 23),
                    minute=random.randint(0, 59)
                )).isoformat()
            }
            bots.append(bot)
        
        # Create bot status summary
        summary = f"You have {len(bots)} active trading bot{'s' if len(bots) != 1 else ''}:\n\n"
        
        for bot in bots:
            name = bot.get("name")
            status = bot.get("status")
            symbol = bot.get("symbol")
            pnl = bot.get("pnl", 0)
            pnl_pct = bot.get("pnl_percentage", 0)
            
            status_emoji = "🟢" if status == "running" else "🔴" if status == "stopped" else "⚠️"
            pnl_str = f"(${pnl:+,.2f}, {pnl_pct:+.2f}%)" if pnl is not None else ""
            
            summary += f"{status_emoji} {name}: Trading {symbol}, {status} {pnl_str}\n"
        
        return {
            "message": summary,
            "data": {
                "bots": bots,
                "action": "show_bots",
                "source": "mock"
            }
        }
    
    def _handle_market_analysis(self, entities: List[Dict[str, Any]], user_id: str, message: str) -> Dict[str, Any]:
        """Handle market_analysis intent"""
        # Extract asset entity
        asset = None
        for entity in entities:
            if entity.get("entity") == "asset":
                asset = entity.get("value")
        
        if not asset:
            return {
                "message": "Which cryptocurrency or stock would you like me to analyze?",
                "data": None
            }
        
        try:
            response = requests.get(
                f"{config.MODEL_SERVICE_URL}/analysis",
                params={"symbol": asset},
                timeout=10  # Analysis might take longer
            )
            
            if response.status_code == 200:
                analysis = response.json()
                
                # Format analysis
                prediction = analysis.get("prediction", {})
                trend = prediction.get("trend")
                confidence = prediction.get("confidence", 0) * 100
                price_target = prediction.get("price_target")
                timeframe = prediction.get("timeframe", "short-term")
                
                indicators = analysis.get("indicators", {})
                rsi = indicators.get("rsi")
                macd = indicators.get("macd")
                
                # Create analysis message
                if trend == "bullish":
                    trend_emoji = "📈"
                    trend_description = "bullish (upward)"
                elif trend == "bearish":
                    trend_emoji = "📉"
                    trend_description = "bearish (downward)"
                else:
                    trend_emoji = "↔️"
                    trend_description = "neutral (sideways)"
                
                analysis_message = f"# Market Analysis for {asset} {trend_emoji}\n\n"
                
                analysis_message += f"Our AI models predict a **{trend_description}** trend with {confidence:.1f}% confidence "
                analysis_message += f"for the {timeframe} timeframe.\n\n"
                
                if price_target:
                    analysis_message += f"Price target: ${price_target:,.2f}\n\n"
                
                analysis_message += "## Technical Indicators\n\n"
                
                if rsi is not None:
                    rsi_status = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
                    analysis_message += f"• RSI: {rsi:.1f} ({rsi_status})\n"
                
                if macd is not None:
                    macd_signal = macd.get("signal", 0)
                    macd_histogram = macd.get("histogram", 0)
                    macd_status = "bullish" if macd_histogram > 0 else "bearish"
                    analysis_message += f"• MACD: {macd_status} (histogram: {macd_histogram:.2f})\n"
                
                # Add market sentiment if available
                sentiment = analysis.get("sentiment")
                if sentiment:
                    analysis_message += f"\n## Market Sentiment\n\n"
                    analysis_message += f"• Social media: {sentiment.get('social', 'neutral')}\n"
                    analysis_message += f"• News: {sentiment.get('news', 'neutral')}\n"
                
                # Add recommendation
                analysis_message += f"\n## Recommendation\n\n"
                recommendation = analysis.get("recommendation", "hold")
                analysis_message += f"Based on our analysis, we suggest to **{recommendation}** {asset}."
                
                return {
                    "message": analysis_message,
                    "data": analysis
                }
            else:
                # Fallback to mock data
                logger.warning(f"Failed to get market analysis: {response.status_code}")
                return self._mock_market_analysis(asset)
                
        except requests.RequestException as e:
            logger.error(f"Error connecting to model service: {e}")
            return self._mock_market_analysis(asset)
    
    def _mock_market_analysis(self, asset: str) -> Dict[str, Any]:
        """Generate mock market analysis for demo purposes"""
        import random
        
        # Generate random analysis
        trends = ["bullish", "bearish", "neutral"]
        trend_weights = [0.4, 0.3, 0.3]  # Slightly biased towards bullish
        trend = random.choices(trends, weights=trend_weights, k=1)[0]
        
        confidence = round(random.uniform(0.65, 0.95), 2)
        
        # Base price on asset
        base_prices = {
            "Bitcoin": 55000,
            "Ethereum": 3500,
            "Litecoin": 200,
            "Ripple": 1.0,
            "Cardano": 2.0,
            "Tesla": 800,
            "Apple": 175,
            "Amazon": 3200,
            "Google": 2700,
            "Microsoft": 325
        }
        
        base_price = base_prices.get(asset, 100)
        
        # Generate price target based on trend
        if trend == "bullish":
            price_change_pct = random.uniform(0.05, 0.15)  # 5-15% increase
        elif trend == "bearish":
            price_change_pct = random.uniform(-0.15, -0.05)  # 5-15% decrease
        else:
            price_change_pct = random.uniform(-0.03, 0.03)  # -3% to +3%
            
        price_target = round(base_price * (1 + price_change_pct), 2)
        
        # Generate RSI based on trend
        if trend == "bullish":
            rsi = random.uniform(55, 75)
        elif trend == "bearish":
            rsi = random.uniform(25, 45)
        else:
            rsi = random.uniform(45, 55)
            
        # Generate MACD based on trend
        if trend == "bullish":
            macd_histogram = random.uniform(0.1, 2.0)
        elif trend == "bearish":
            macd_histogram = random.uniform(-2.0, -0.1)
        else:
            macd_histogram = random.uniform(-0.5, 0.5)
            
        macd = {
            "value": random.uniform(-5, 5),
            "signal": random.uniform(-5, 5),
            "histogram": macd_histogram
        }
        
        # Generate sentiment
        sentiment_options = ["very negative", "negative", "neutral", "positive", "very positive"]
        if trend == "bullish":
            sentiment_weights = [0.05, 0.1, 0.2, 0.4, 0.25]
        elif trend == "bearish":
            sentiment_weights = [0.25, 0.4, 0.2, 0.1, 0.05]
        else:
            sentiment_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
            
        social_sentiment = random.choices(sentiment_options, weights=sentiment_weights, k=1)[0]
        news_sentiment = random.choices(sentiment_options, weights=sentiment_weights, k=1)[0]
        
        # Generate recommendation
        if trend == "bullish" and confidence > 0.8:
            recommendation = "buy"
        elif trend == "bearish" and confidence > 0.8:
            recommendation = "sell"
        else:
            recommendation = "hold"
        
        # Create analysis data
        analysis = {
            "asset": asset,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": {
                "trend": trend,
                "confidence": confidence,
                "price_target": price_target,
                "timeframe": random.choice(["short-term", "medium-term", "long-term"])
            },
            "indicators": {
                "rsi": rsi,
                "macd": macd
            },
            "sentiment": {
                "social": social_sentiment,
                "news": news_sentiment
            },
            "recommendation": recommendation,
            "source": "mock"
        }
        
        # Create analysis message
        if trend == "bullish":
            trend_emoji = "📈"
            trend_description = "bullish (upward)"
        elif trend == "bearish":
            trend_emoji = "📉"
            trend_description = "bearish (downward)"
        else:
            trend_emoji = "↔️"
            trend_description = "neutral (sideways)"
        
        analysis_message = f"# Market Analysis for {asset} {trend_emoji}\n\n"
        
        analysis_message += f"Our AI models predict a **{trend_description}** trend with {confidence*100:.1f}% confidence "
        analysis_message += f"for the {analysis['prediction']['timeframe']} timeframe.\n\n"
        
        analysis_message += f"Price target: ${price_target:,.2f}\n\n"
        
        analysis_message += "## Technical Indicators\n\n"
        
        rsi_status = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        analysis_message += f"• RSI: {rsi:.1f} ({rsi_status})\n"
        
        macd_status = "bullish" if macd_histogram > 0 else "bearish"
        analysis_message += f"• MACD: {macd_status} (histogram: {macd_histogram:.2f})\n"
        
        # Add market sentiment
        analysis_message += f"\n## Market Sentiment\n\n"
        analysis_message += f"• Social media: {social_sentiment}\n"
        analysis_message += f"• News: {news_sentiment}\n"
        
        # Add recommendation
        analysis_message += f"\n## Recommendation\n\n"
        analysis_message += f"Based on our analysis, we suggest to **{recommendation}** {asset}."
        
        return {
            "message": analysis_message,
            "data": analysis
        }
