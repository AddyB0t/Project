from typing import List, Dict, Any, Optional
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv

from services.embedding_service import EmbeddingService
from services.technical_content_processor import get_technical_processor
from models.embedding import SimilaritySearchResult, SimilaritySearchResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    """Model for chat requests with context"""
    message: str
    use_context: bool = True
    max_context_chunks: int = 8
    similarity_threshold: float = 0.3
    document_name: Optional[str] = None
    conversation_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    """Model for chat responses"""
    success: bool
    response: str
    context_used: List[SimilaritySearchResult] = []
    search_time: float = 0.0
    total_response_time: float = 0.0
    message: str = ""

class ChatbotService:
    """Service for chatbot with document context using similarity search"""
    
    def __init__(self):
        """Initialize the chatbot service with embedding service"""
        try:
            self.embedding_service = EmbeddingService()
            
            # Define greeting patterns for classification
            self.greeting_patterns = {
                'hi', 'hello', 'hey', 'howdy', 'greetings', 'good morning', 
                'good afternoon', 'good evening', 'how are you', 'whats up', 
                'what\'s up', 'hiya', 'sup', 'yo', 'heya', 'good day',
                'thanks', 'thank you', 'okay', 'ok', 'i see', 'got it',
                'cool', 'nice', 'awesome', 'great', 'perfect', 'sounds good',
                'help', 'can you help', 'please help', 'assist me',
                'what can you do', 'how do you work', 'what are your capabilities',
                'good night', 'bye', 'goodbye', 'see you', 'catch you later',
                'please', 'excuse me', 'pardon me', 'sorry'
            }
            
            logger.info("ChatbotService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotService: {str(e)}")
            raise
    
    def is_greeting(self, query: str) -> bool:
        """
        Check if the user query is a greeting
        
        Args:
            query: User's input query
            
        Returns:
            True if the query appears to be a greeting, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Remove punctuation for better matching
        import re
        clean_query = re.sub(r'[^\w\s]', '', query_lower)
        
        # Check if the entire query is a greeting
        if clean_query in self.greeting_patterns:
            return True
        
        # Check if query starts with a greeting (for cases like "hi there")
        for greeting in self.greeting_patterns:
            if clean_query.startswith(greeting + ' ') or clean_query == greeting:
                return True
        
        return False
    
    def classify_query_type(self, query: str) -> str:
        """
        Classify the type of user query
        
        Args:
            query: User's input query
            
        Returns:
            Query type: 'greeting', 'document', or 'general'
        """
        if self.is_greeting(query):
            return 'greeting'
        
        # Simple heuristics for document-related queries
        document_indicators = [
            'what', 'how', 'when', 'where', 'why', 'who', 'explain', 'describe',
            'tell me', 'show me', 'find', 'search', 'document', 'file', 'content',
            'information', 'details', 'summary', 'overview'
        ]
        
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in document_indicators):
            return 'document'
        
        # Default to general for other queries
        return 'general'
    
    def should_use_hybrid_mode(self, query: str) -> bool:
        """
        Determine if hybrid mode should be used for this query
        
        Args:
            query: User's input query
            
        Returns:
            True if hybrid mode should be used, False for strict document mode
        """
        query_type = self.classify_query_type(query)
        
        # Use hybrid mode for all query types by default to allow natural conversation
        # Only use strict document mode for very specific document analysis requests
        specific_doc_keywords = ['analyze this document', 'summarize this document', 'extract from document']
        query_lower = query.lower()
        
        is_specific_doc_request = any(keyword in query_lower for keyword in specific_doc_keywords)
        
        # Default to hybrid mode unless it's a very specific document analysis request
        return not is_specific_doc_request
    
    def search_relevant_context(
        self, 
        query: str, 
        max_chunks: int = 8, 
        similarity_threshold: float = 0.3,
        document_name: str = None,
        document_id: str = None
    ) -> SimilaritySearchResponse:
        """
        Search for relevant document context using similarity search
        
        Args:
            query: User query to search for
            max_chunks: Maximum number of context chunks to retrieve (k=8 by default)
            similarity_threshold: Minimum similarity score threshold
            document_name: Optional document name for filtering (legacy)
            document_id: Optional document ID for filtering (preferred)
            
        Returns:
            SimilaritySearchResponse with relevant documents
        """
        try:
            logger.info(f"Searching for relevant context: k={max_chunks}, threshold={similarity_threshold}")
            if document_id:
                logger.info(f"FILTERING BY DOCUMENT_ID: '{document_id}'")
            elif document_name:
                logger.info(f"FILTERING BY DOCUMENT_NAME: '{document_name}'")
            
            # Priority: Use document_id filtering if available (most precise)
            if document_id:
                # Search within specific document using document_id
                search_results = self.embedding_service.search_similar_documents_by_id(
                    query=query,
                    document_id=document_id,
                    limit=max_chunks,
                    similarity_threshold=similarity_threshold
                )
                
                logger.info(f"DOCUMENT_ID SEARCH: Found {len(search_results)} results for document_id='{document_id}'")
                
                # Debug: Log first result if available
                if search_results:
                    logger.info(f"FIRST RESULT: {search_results[0].get('chunk_text', '')[:100]}...")
                
                # Convert to SimilaritySearchResponse format
                results = []
                for result in search_results:
                    results.append(SimilaritySearchResult(
                        document_id=result['document_id'],
                        chunk_id=result['chunk_id'],
                        chunk_text=result['chunk_text'],
                        chunk_index=result['chunk_index'],
                        similarity_score=result['similarity_score'],
                        original_filename=result.get('original_filename', 'unknown'),
                        file_type=result.get('file_type', 'unknown'),
                        metadata=result.get('metadata', {})
                    ))
                
                search_response = SimilaritySearchResponse(
                    success=True,
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_time=0,
                    similarity_threshold=similarity_threshold,
                    message=f"Found {len(results)} relevant chunks from document_id: {document_id}"
                )
                
                logger.info(f"CONTEXT RESPONSE: Created response with {len(results)} chunks")
                
            else:
                # Use the existing similarity search for all documents
                logger.info("SEARCHING ALL DOCUMENTS (no filter)")
                search_response = self.embedding_service.search_similar_documents(
                    query=query,
                    limit=max_chunks,
                    similarity_threshold=similarity_threshold
                )
                logger.info(f"ALL-DOCUMENT SEARCH: Found {len(search_response.results)} results")
            
            logger.info(f"FINAL CONTEXT: Returning {len(search_response.results)} context chunks")
            return search_response
            
        except Exception as e:
            logger.error(f"Error searching for relevant context: {str(e)}")
            return SimilaritySearchResponse(
                success=False,
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                similarity_threshold=similarity_threshold,
                message=f"Search failed: {str(e)}"
            )
    
    def build_context_prompt(
        self, 
        user_query: str, 
        context_results: List[SimilaritySearchResult],
        conversation_history: List[ChatMessage] = None
    ) -> str:
        """
        Build a comprehensive prompt with document context for the LLM
        
        Args:
            user_query: The user's current question
            context_results: Relevant document chunks from similarity search
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted prompt string for the LLM
        """
        try:
            if not context_results:
                logger.info("NO CONTEXT: Returning original message")
                return f"User Question: {user_query}"
            
            logger.info(f"FORMATTING PROMPT with {len(context_results)} context chunks")
            
            prompt_parts = []
            
            # Add document context if available
            if context_results:
                prompt_parts.append("Based on the following document information:")
                prompt_parts.append("")
                
                for i, result in enumerate(context_results, 1):
                    # Enhance technical content for better LLM understanding
                    content_text = result.chunk_text
                    
                    # Check if this appears to be technical drawing content
                    if self._is_technical_content(result.original_filename, result.chunk_text):
                        try:
                            technical_processor = get_technical_processor()
                            enhanced_content = technical_processor.enhance_chunk_content(result.chunk_text)
                            content_text = enhanced_content
                            logger.info(f"TECHNICAL PROCESSING: Enhanced chunk {i} from {len(result.chunk_text)} to {len(enhanced_content)} chars")
                        except Exception as tech_error:
                            logger.warning(f"Technical processing failed for chunk {i}: {str(tech_error)}")
                            # Fall back to original content
                    
                    context_section = f"""Document: {result.original_filename}
Content: {content_text}
Relevance Score: {result.similarity_score:.3f}
"""
                    prompt_parts.append(context_section)
                    logger.info(f"CONTEXT CHUNK {i}: Added {len(content_text)} chars from {result.original_filename}")
                
                prompt_parts.append("=== END CONTEXT ===\n")
            
            # Add conversation history if available
            if conversation_history:
                prompt_parts.append("\n=== CONVERSATION HISTORY ===")
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    prompt_parts.append(f"{msg.role.upper()}: {msg.content}")
                prompt_parts.append("=== END HISTORY ===\n")
            
            # Add the current user query
            prompt_parts.append(f"\nUser Question: {user_query}")
            prompt_parts.append("\nPlease provide a helpful response based on the available context and your knowledge. When referencing information from the documents, please mention which document you are citing (e.g., 'According to [document name]...' or 'Based on the [document name] document...'):")
            
            final_prompt = "\n".join(prompt_parts)
            logger.info(f"FINAL PROMPT: {len(final_prompt)} characters total")
            logger.info(f"PROMPT PREVIEW: {final_prompt[:200]}...")
            
            return final_prompt
            
        except Exception as e:
            logger.error(f"Error building context prompt: {str(e)}")
            return f"User Question: {user_query}\n\nPlease provide a helpful response:"
    
    def _is_technical_content(self, filename: str, content: str) -> bool:
        """
        Determine if content should be processed as technical drawing content
        
        Args:
            filename: Original filename
            content: Text content to check
            
        Returns:
            True if content appears to be technical drawing data
        """
        import re
        
        # Check filename for technical indicators
        filename_lower = filename.lower() if filename else ""
        filename_indicators = ['drawing', 'plan', 'blueprint', 'technical', 'architectural', 'civil']
        
        # Check content for technical patterns
        content_lower = content.lower()
        
        # Technical content indicators
        has_many_numbers = len([c for c in content if c.isdigit()]) > len(content) * 0.3  # More than 30% digits
        has_coordinates = bool(re.search(r'\b\d{3,5}\s+\d{3,5}\b', content))  # Coordinate patterns
        has_room_labels = bool(re.search(r'\b(OFFICE|ROOM|RECEPTION|PASSAGE)\b', content, re.IGNORECASE))
        has_measurements = bool(re.search(r'\b\d+\s*mm\b', content, re.IGNORECASE))
        
        # Determine if technical
        is_technical = (
            any(indicator in filename_lower for indicator in filename_indicators) or
            (has_many_numbers and (has_coordinates or has_room_labels or has_measurements))
        )
        
        if is_technical:
            logger.info(f"Detected technical content in {filename}: numbers={has_many_numbers}, coordinates={has_coordinates}, rooms={has_room_labels}")
        
        return is_technical
    
    def format_context_summary(self, context_results: List[SimilaritySearchResult]) -> str:
        """
        Create a summary of the context sources used
        
        Args:
            context_results: List of context chunks used
            
        Returns:
            Formatted summary string
        """
        if not context_results:
            return "No document context used."
        
        sources = {}
        for result in context_results:
            filename = result.original_filename
            if filename not in sources:
                sources[filename] = {
                    'chunks': 0,
                    'avg_similarity': 0.0,
                    'similarities': []
                }
            sources[filename]['chunks'] += 1
            sources[filename]['similarities'].append(result.similarity_score)
        
        # Calculate average similarities
        for source in sources.values():
            source['avg_similarity'] = sum(source['similarities']) / len(source['similarities'])
        
        summary_parts = [f"Used context from {len(context_results)} document chunks:"]
        for filename, info in sources.items():
            summary_parts.append(
                f"- {filename}: {info['chunks']} chunks "
                f"(avg similarity: {info['avg_similarity']:.3f})"
            )
        
        return "\\n".join(summary_parts)
    
    def prepare_chat_response(
        self, 
        request: ChatRequest
    ) -> tuple[str, List[SimilaritySearchResult], float]:
        """
        Prepare the chat response with context
        
        Args:
            request: ChatRequest with user message and settings
            
        Returns:
            Tuple of (formatted_prompt, context_results, search_time)
        """
        start_time = time.time()
        context_results = []
        search_time = 0.0
        
        try:
            if request.use_context:
                logger.info(f"PREPARING CONTEXT for query: '{request.message[:50]}...'")
                logger.info(f"DOCUMENT FILTER: {request.document_name}")
                
                # Search for relevant context
                search_start = time.time()
                search_response = self.search_relevant_context(
                    query=request.message,
                    max_chunks=request.max_context_chunks,
                    similarity_threshold=request.similarity_threshold,
                    document_name=request.document_name
                )
                search_time = time.time() - search_start
                
                if search_response.success:
                    context_results = search_response.results
                    logger.info(f"CONTEXT PREPARED: {len(context_results)} chunks retrieved")
                    
                    # Debug: Log context content
                    for i, result in enumerate(context_results[:2]):  # Log first 2 chunks
                        logger.info(f"CHUNK {i+1}: {result.chunk_text[:100]}...")
                        
                else:
                    logger.warning(f"CONTEXT SEARCH FAILED: {search_response.message}")
            else:
                logger.info("CONTEXT DISABLED: use_context=False")
            
            # Build the prompt with context
            formatted_prompt = self.build_context_prompt(
                user_query=request.message,
                context_results=context_results,
                conversation_history=request.conversation_history
            )
            
            logger.info(f"PROMPT FORMATTED: {len(formatted_prompt)} characters")
            
            return formatted_prompt, context_results, search_time
            
        except Exception as e:
            logger.error(f"Error preparing chat response: {str(e)}")
            # Return basic prompt without context on error
            return f"User Question: {request.message}", [], 0.0
    
    def get_context_metadata(self, context_results: List[SimilaritySearchResult]) -> Dict[str, Any]:
        """
        Extract metadata about the context used
        
        Args:
            context_results: List of context chunks
            
        Returns:
            Dictionary with context metadata
        """
        if not context_results:
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "avg_similarity": 0.0,
                "sources": []
            }
        
        unique_docs = set()
        similarities = []
        sources = []
        
        for result in context_results:
            unique_docs.add(result.document_id)
            similarities.append(result.similarity_score)
            sources.append({
                "filename": result.original_filename,
                "similarity": result.similarity_score,
                "chunk_index": result.chunk_index
            })
        
        return {
            "total_chunks": len(context_results),
            "unique_documents": len(unique_docs),
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
            "sources": sources
        }
    
    def create_chat_response_for_frontend(
        self,
        user_message: str,
        use_context: bool = True,
        max_context_chunks: int = 8,
        similarity_threshold: float = 0.3,
        document_name: str = None,
        conversation_history: List[ChatMessage] = None
    ) -> Dict[str, Any]:
        """
        Create a response optimized for frontend consumption
        
        Args:
            user_message: User's message
            use_context: Whether to use document context
            max_context_chunks: Maximum chunks to retrieve (k=8)
            similarity_threshold: Similarity threshold
            conversation_history: Previous conversation
            
        Returns:
            Dictionary with response data for frontend
        """
        start_time = time.time()
        
        try:
            request = ChatRequest(
                message=user_message,
                use_context=use_context,
                max_context_chunks=max_context_chunks,
                similarity_threshold=similarity_threshold,
                document_name=document_name,
                conversation_history=conversation_history or []
            )
            
            # Prepare the response with context
            formatted_prompt, context_results, search_time = self.prepare_chat_response(request)
            
            # Get context metadata
            context_metadata = self.get_context_metadata(context_results)
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "formatted_prompt": formatted_prompt,
                "context_summary": self.format_context_summary(context_results),
                "context_metadata": context_metadata,
                "search_time": search_time,
                "total_preparation_time": total_time,
                "context_chunks": [
                    {
                        "document_id": r.document_id,
                        "filename": r.original_filename,
                        "content": r.chunk_text,
                        "similarity": r.similarity_score,
                        "chunk_index": r.chunk_index
                    }
                    for r in context_results
                ],
                "message": f"Prepared response with {len(context_results)} context chunks"
            }
            
        except Exception as e:
            logger.error(f"Error creating chat response for frontend: {str(e)}")
            return {
                "success": False,
                "formatted_prompt": f"User Question: {user_message}",
                "context_summary": "Error retrieving context",
                "context_metadata": {"total_chunks": 0, "unique_documents": 0},
                "search_time": 0.0,
                "total_preparation_time": 0.0,
                "context_chunks": [],
                "message": f"Error: {str(e)}"
            }