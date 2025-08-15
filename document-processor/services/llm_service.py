import os
import httpx
import logging
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterLLMService:
    """Service for interacting with OpenRouter API using GPT-3.5-turbo"""
    
    def __init__(self):
        """Initialize the OpenRouter LLM service"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "openai/gpt-3.5-turbo"
        
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables")
            raise ValueError("OpenRouter API key is required")
        
        logger.info(f"OpenRouter LLM service initialized with model: {self.model}")
    
    DOCUMENT_RESTRICTED_PROMPT = """You are a helpful document assistant. Answer questions using the provided document context below.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Use the document content above to answer the question
- If you can find relevant information, provide a helpful answer based on the document content
- If specific details are missing, explain what information IS available in the document
- Be specific and cite relevant parts of the documents when possible
- Only say "I cannot find that information" if the document context is completely empty or unrelated to the question
- Keep your responses concise and focused on the document content

Please provide your response based on the document context:"""

    HYBRID_PROMPT = """You are a helpful and friendly assistant. You can help with document questions and also provide general assistance.

GUIDELINES:
- For greetings and casual conversation, respond warmly and naturally
- For document-related questions, prioritize information from the provided document context
- If document context is available and relevant, use it to provide detailed answers with citations
- If no relevant document context is available, you can still provide helpful general knowledge
- For general questions, provide thoughtful and useful responses
- Be conversational, helpful, and informative
- Keep responses clear and appropriately detailed

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

Please provide a helpful response:"""

    async def ask_question_with_context(
        self, 
        user_question: str, 
        document_context: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        use_hybrid_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question to the LLM with document context restriction
        
        Args:
            user_question: The user's question
            document_context: Relevant document context for the question
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0, lower = more focused)
            use_hybrid_mode: If True, allows greetings and general conversation. If False, strict document-only mode
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Choose prompt template based on mode
            prompt_template = self.HYBRID_PROMPT if use_hybrid_mode else self.DOCUMENT_RESTRICTED_PROMPT
            
            # Format the prompt with context and question
            formatted_prompt = prompt_template.format(
                context=document_context.strip() if document_context else "No document context provided.",
                question=user_question.strip()
            )
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful and friendly assistant. When documents are provided, prioritize that information, but you can also help with general questions and conversation."
                    },
                    {
                        "role": "user", 
                        "content": formatted_prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",  # Optional: for OpenRouter analytics
                "X-Title": "Document Assistant"  # Optional: for OpenRouter analytics
            }
            
            logger.info(f"Sending request to OpenRouter API for question: {user_question[:50]}...")
            logger.debug(f"Using model: {self.model}")
            logger.debug(f"Context length: {len(document_context)} characters")
            
            # Debug: Log the final prompt being sent to LLM
            logger.info(f"🔍 LLM SERVICE DEBUG:")
            logger.info(f"  Use hybrid mode: {use_hybrid_mode}")
            logger.info(f"  Final prompt preview (first 500 chars): {formatted_prompt[:500]}...")
            if len(formatted_prompt) > 500:
                logger.info(f"  Final prompt preview (last 300 chars): ...{formatted_prompt[-300:]}")
            
            # Make the API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                logger.info(f"OpenRouter API response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract the response text
                    llm_response = result["choices"][0]["message"]["content"]
                    
                    # Get usage information
                    usage = result.get("usage", {})
                    
                    logger.info(f"LLM response generated successfully")
                    logger.debug(f"Tokens used - Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)}")
                    
                    return {
                        "success": True,
                        "response": llm_response.strip(),
                        "model_used": self.model,
                        "usage": usage,
                        "has_context": bool(document_context and document_context.strip()),
                        "context_length": len(document_context) if document_context else 0,
                        "message": "Response generated successfully"
                    }
                else:
                    error_detail = response.text
                    logger.error(f"OpenRouter API error {response.status_code}: {error_detail}")
                    
                    return {
                        "success": False,
                        "response": "I'm sorry, I'm currently unable to process your request. Please try again later.",
                        "error": f"API Error {response.status_code}: {error_detail}",
                        "message": "Failed to get response from LLM service"
                    }
                    
        except httpx.TimeoutException:
            logger.error("OpenRouter API request timed out")
            return {
                "success": False,
                "response": "The request timed out. Please try again with a shorter question.",
                "error": "Request timeout",
                "message": "LLM service request timed out"
            }
            
        except Exception as e:
            logger.error(f"Error in OpenRouter LLM service: {str(e)}")
            return {
                "success": False,
                "response": "I'm sorry, I encountered an error while processing your request.",
                "error": str(e),
                "message": "LLM service error"
            }
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is properly configured
        
        Returns:
            True if API key is available, False otherwise
        """
        return bool(self.api_key and self.api_key.startswith("sk-or-"))
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the current model configuration
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "provider": "OpenRouter",
            "api_endpoint": f"{self.base_url}/chat/completions",
            "api_key_configured": self.validate_api_key()
        }

# Global instance
_llm_service_instance = None

def get_llm_service() -> Optional[OpenRouterLLMService]:
    """
    Get or create a singleton instance of the OpenRouter LLM service
    
    Returns:
        OpenRouterLLMService instance or None if initialization fails
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        try:
            _llm_service_instance = OpenRouterLLMService()
            logger.info("OpenRouter LLM service singleton created")
        except Exception as e:
            logger.error(f"Failed to create OpenRouter LLM service: {str(e)}")
            return None
    
    return _llm_service_instance