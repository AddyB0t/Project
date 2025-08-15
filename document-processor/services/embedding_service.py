import os
import sys
import time
import json
import uuid
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Use sentence-transformers directly for better control
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing document embeddings using sentence-transformers and Supabase"""
    
    def __init__(self):
        
        # Environment check
        
        try:
            # Initialize sentence transformer directly
            model_name = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
            
            self.model = SentenceTransformer(model_name)
            
            # Configuration
            self.chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", 1000))
            self.chunk_overlap = int(os.getenv("EMBEDDING_CHUNK_OVERLAP", 200))
            self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", 768))
            
            
        except Exception as e:
            raise
        
        # Initialize Supabase client with SERVICE_KEY to bypass RLS for embeddings
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url:
                raise ValueError("SUPABASE_URL environment variable not set")
            if not supabase_service_key:
                raise ValueError("SUPABASE_SERVICE_KEY environment variable not set")
            
            
            self.supabase = create_client(supabase_url, supabase_service_key)
            
            # Test connection by trying to access a table
            try:
                response = self.supabase.table('document_embeddings').select("count", count="exact").limit(0).execute()
            except Exception as test_e:
                pass
            
        except Exception as e:
            raise
        
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            })
            
            start += chunk_size - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def create_embeddings(self, document_id: str, text: str, document_name: str = None) -> int:
        """Create embeddings for document text and store in Supabase"""
        
        # Get document name if not provided
        if document_name is None:
            document_name = self._get_document_name(document_id)
        
        # Text chunking
        chunks = self.chunk_text(text)
        
        embeddings_created = 0
        batch_size = 10  # Process in batches
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            batch_data = []
            
            for j, chunk in enumerate(batch):
                # Generate embedding using sentence-transformers
                embedding_vector = self.model.encode(chunk['text'])
                
                # Ensure proper format for pgvector (convert to list if needed)
                if hasattr(embedding_vector, 'tolist'):
                    embedding_list = embedding_vector.tolist()
                else:
                    embedding_list = list(embedding_vector)
                
                # Validate embedding dimensions
                if len(embedding_list) != 768:
                    raise ValueError(f"Embedding dimension mismatch: expected 768, got {len(embedding_list)}")
                
                # Prepare data for insertion
                chunk_data = {
                    'document_id': document_id,
                    'document_name': document_name,
                    'chunk_id': f"{document_id}_chunk_{i + j}",
                    'chunk_text': chunk['text'],
                    'chunk_index': i + j,
                    'embedding': embedding_list,
                    'metadata': {
                        'word_count': chunk['word_count'],
                        'char_count': chunk['char_count']
                    }
                }
                batch_data.append(chunk_data)
            
            # Insert batch into Supabase
            try:
                result = self.supabase.table('document_embeddings').insert(batch_data).execute()
                
                if result.data:
                    embeddings_created += len(batch_data)
                    logger.info(f"Successfully inserted batch {i//batch_size + 1} ({len(batch_data)} embeddings)")
                else:
                    logger.error(f"Failed to insert batch {i//batch_size + 1}: No data returned from Supabase")
                    raise Exception(f"No data returned from Supabase for batch {i//batch_size + 1}")
                    
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {str(e)}")
                raise
        
        logger.info(f"Embedding creation completed for document {document_id}: {embeddings_created} total embeddings")
        return embeddings_created
    
    def get_embedding_stats(self):
        """Get statistics about stored embeddings by counting actual embeddings"""
        try:
            
            # Count actual embeddings in document_embeddings table
            embeddings_response = self.supabase.table("document_embeddings").select("document_id, chunk_text").execute()
            
            if embeddings_response.data:
                total_chunks = len(embeddings_response.data)
                
                # Count unique documents
                unique_documents = set()
                total_characters = 0
                
                for row in embeddings_response.data:
                    unique_documents.add(row['document_id'])
                    if row.get('chunk_text'):
                        total_characters += len(row['chunk_text'])
                
                total_documents = len(unique_documents)
                avg_chunks = total_chunks / total_documents if total_documents > 0 else 0
                
            else:
                total_documents = total_chunks = total_characters = avg_chunks = 0
            
            stats = {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_characters": total_characters,
                "average_chunks_per_document": round(avg_chunks, 2),
                "embedding_dimensions": self.embedding_dimensions,
                "last_updated": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            # Return default stats when there's an error
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "average_chunks_per_document": 0,
                "embedding_dimensions": self.embedding_dimensions,
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def search_similar_documents(self, query: str, limit: int = 5, similarity_threshold: float = 0.5):
        """Search for similar documents using embeddings with proper result format"""
        try:
            
            # Generate embedding for the query
            query_embedding = self.model.encode(query).tolist()
            
            # Search for similar embeddings in Supabase
            embeddings_response = self.supabase.table("document_embeddings").select(
                "document_id, document_name, chunk_id, chunk_text, chunk_index, metadata, embedding"
            ).execute()
            
            if not embeddings_response.data:
                return self._create_empty_search_response(query, similarity_threshold)
            
            
            # Calculate similarities manually (since we don't have pgvector functions set up)
            import numpy as np
            results = []
            
            for row in embeddings_response.data:
                try:
                    # Get the stored embedding
                    if 'embedding' not in row or not row['embedding']:
                        continue
                    
                    stored_embedding = row['embedding']
                    
                    # Parse embedding if it's stored as a JSON string
                    if isinstance(stored_embedding, str):
                        try:
                            stored_embedding = json.loads(stored_embedding)
                        except json.JSONDecodeError:
                            continue
                    
                    # Validate embedding is a list/array
                    if not isinstance(stored_embedding, (list, tuple)):
                        continue
                        
                    if len(stored_embedding) == 0:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= similarity_threshold:
                        result = {
                            "document_id": row['document_id'],
                            "chunk_id": row['chunk_id'],
                            "chunk_text": row['chunk_text'],
                            "chunk_index": row['chunk_index'],
                            "similarity_score": float(similarity),
                            "original_filename": row.get('document_name', 'unknown'),
                            "file_type": "unknown",
                            "metadata": row.get('metadata', {})
                        }
                        results.append(result)
                        
                except Exception as embed_error:
                    continue
            
            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit results
            results = results[:limit]
            
            
            return type('SearchResponse', (), {
                'success': True,
                'query': query,
                'results': [type('Result', (), r)() for r in results],
                'total_results': len(results),
                'search_time': 0.0,
                'similarity_threshold': similarity_threshold,
                'message': f"Found {len(results)} similar documents"
            })()
            
        except Exception as e:
            return self._create_empty_search_response(query, similarity_threshold, str(e))
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors with robust error handling"""
        import numpy as np
        
        try:
            # Convert to numpy arrays
            a = np.array(a, dtype=np.float32)
            b = np.array(b, dtype=np.float32)
            
            # Check if arrays are valid (not scalar)
            if a.ndim == 0 or b.ndim == 0:
                return 0.0
            
            # Check if arrays have the same length
            if a.shape != b.shape:
                return 0.0
            
            # Check for empty arrays
            if a.size == 0 or b.size == 0:
                return 0.0
            
            # Calculate norms
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            # Check for zero vectors
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            similarity = dot_product / (norm_a * norm_b)
            
            # Ensure similarity is in valid range [-1, 1]
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def _create_empty_search_response(self, query, similarity_threshold, error=None):
        """Create empty search response"""
        return type('SearchResponse', (), {
            'success': False if error else True,
            'query': query,
            'results': [],
            'total_results': 0,
            'search_time': 0.0,
            'similarity_threshold': similarity_threshold,
            'message': error if error else "No similar documents found"
        })()
    
    def _get_document_name(self, document_id: str) -> str:
        """Get document name from metadata table"""
        try:
            result = self.supabase.table('documents_metadata')\
                .select('original_filename')\
                .eq('document_id', document_id)\
                .single()\
                .execute()
            
            if result.data and result.data.get('original_filename'):
                return result.data['original_filename']
            else:
                return f"Document_{document_id[:8]}"  # Fallback name
                
        except Exception as e:
            logger.warning(f"Could not get document name for {document_id}: {str(e)}")
            return f"Document_{document_id[:8]}"  # Fallback name
    
    def get_available_document_names(self):
        """Get list of unique documents with document_id and names for dropdown"""
        try:
            # Query document_embeddings table for unique document_id and document_name pairs
            result = self.supabase.table('document_embeddings')\
                .select('document_id, document_name')\
                .execute()
            
            if result.data:
                # Group by document_id to get unique documents
                document_map = {}
                for row in result.data:
                    doc_id = row.get('document_id')
                    doc_name = row.get('document_name')
                    
                    if doc_id and doc_name and doc_name.strip():
                        document_map[doc_id] = doc_name
                
                # Convert to list and sort by document name for better UX
                documents = [{'id': doc_id, 'name': doc_name} for doc_id, doc_name in document_map.items()]
                documents.sort(key=lambda x: x['name'])
                
                logger.info(f"Found {len(documents)} unique documents with IDs")
                return documents
            else:
                logger.warning("No document names found in embeddings table")
                return []
                
        except Exception as e:
            logger.error(f"Error getting available document names: {str(e)}")
            return []
    
    def search_similar_documents_by_name(self, query: str, limit: int = 5, 
                                       similarity_threshold: float = 0.5, 
                                       document_name: str = None):
        """Search similar documents with optional document name filtering"""
        try:
            logger.info(f"=== EMBEDDING SEARCH START ===")
            logger.info(f"SEARCH PARAMS: query='{query[:50]}...', limit={limit}")
            logger.info(f"SEARCH PARAMS: threshold={similarity_threshold}, document_name='{document_name}'")
            
            # Generate query embedding
            logger.info("ENCODING: Generating query embedding...")
            query_embedding = self.model.encode(query)
            logger.info(f"ENCODING: Generated embedding with shape {query_embedding.shape}")
            
            # Convert to list format for Supabase
            if hasattr(query_embedding, 'tolist'):
                embedding_list = query_embedding.tolist()
            else:
                embedding_list = list(query_embedding)
            
            logger.info(f"EMBEDDING: Converted to list with {len(embedding_list)} dimensions")
            
            # Use manual similarity search (similar to search_similar_documents method)
            logger.info("DATABASE: Using manual similarity search...")
            try:
                # Get all embeddings and calculate similarity manually (like in search_similar_documents)
                embeddings_response = self.supabase.table("document_embeddings").select(
                    "document_id, document_name, chunk_id, chunk_text, chunk_index, metadata, embedding"
                ).execute()
                
                if not embeddings_response.data:
                    logger.warning("DATABASE: No embeddings found in table")
                    return []
                
                # Calculate similarities manually
                import numpy as np
                import json
                all_results = []
                
                logger.info(f"DATABASE: Processing {len(embeddings_response.data)} embeddings...")
                processed_count = 0
                threshold_passed = 0
                
                for i, row in enumerate(embeddings_response.data):
                    try:
                        # Get the stored embedding
                        if 'embedding' not in row or not row['embedding']:
                            logger.warning(f"DATABASE: Row {i} missing embedding")
                            continue
                        
                        stored_embedding = row['embedding']
                        
                        # Parse embedding if it's stored as a JSON string
                        if isinstance(stored_embedding, str):
                            try:
                                stored_embedding = json.loads(stored_embedding)
                            except json.JSONDecodeError:
                                logger.warning(f"DATABASE: Row {i} invalid JSON embedding")
                                continue
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(embedding_list, stored_embedding)
                        processed_count += 1
                        
                        # Log first few similarities for debugging
                        if i < 5:
                            logger.info(f"DATABASE: Row {i} similarity={similarity:.4f}, threshold={similarity_threshold}, doc_name='{row.get('document_name', 'N/A')}'")
                        
                        if similarity >= similarity_threshold:
                            threshold_passed += 1
                            result = {
                                "document_id": row['document_id'],
                                "chunk_id": row['chunk_id'],
                                "chunk_text": row['chunk_text'],
                                "chunk_index": row['chunk_index'],
                                "similarity_score": float(similarity),
                                "document_name": row.get('document_name', 'unknown'),
                                "original_filename": row.get('document_name', 'unknown'),  # Use document_name as original_filename
                                "filename": row.get('document_name', 'unknown'),  # Also add filename field
                                "file_type": "unknown",
                                "metadata": row.get('metadata', {})
                            }
                            all_results.append(result)
                            
                    except Exception as embed_error:
                        logger.error(f"DATABASE: Error processing row {i}: {str(embed_error)}")
                        continue
                
                logger.info(f"DATABASE: Processed {processed_count} embeddings, {threshold_passed} passed threshold, {len(all_results)} final results")
                
                # Sort by similarity score (highest first)
                all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                result = type('Result', (), {'data': all_results})()
                logger.info(f"DATABASE: Manual search found {len(all_results)} results above threshold")
                
                all_results = result.data if result.data else []
                
                # Filter by document name if provided (use original_filename field)
                if document_name:
                    logger.info(f"FILTERING: Target document name: '{document_name}'")
                    logger.info(f"FILTERING: Before filtering - {len(all_results)} total results")
                    
                    # Debug: Log all available document names from all fields
                    available_docs = set()
                    available_original_filenames = set()
                    available_filenames = set()
                    for r in all_results[:5]:  # Log first 5 for debugging
                        doc_name = r.get('document_name', 'N/A')
                        original_filename = r.get('original_filename', 'N/A')
                        filename = r.get('filename', 'N/A')
                        available_docs.add(doc_name)
                        available_original_filenames.add(original_filename)
                        available_filenames.add(filename)
                        logger.info(f"FILTERING: Result has document_name: '{doc_name}', original_filename: '{original_filename}', filename: '{filename}'")
                    
                    logger.info(f"FILTERING: Available document_names: {sorted(available_docs)}")
                    logger.info(f"FILTERING: Available original_filenames: {sorted(available_original_filenames)}")
                    logger.info(f"FILTERING: Available filenames: {sorted(available_filenames)}")
                    
                    # Perform filtering using both original_filename and filename fields for compatibility
                    search_results = [r for r in all_results if 
                                    r.get('original_filename') == document_name or r.get('filename') == document_name][:limit]
                    logger.info(f"FILTERING: After filtering by original_filename/filename - {len(search_results)} matching results")
                    
                    if len(search_results) == 0:
                        logger.warning(f"FILTERING: NO MATCHES found for document_name='{document_name}'")
                        logger.warning(f"FILTERING: Available original_filenames: {sorted(available_original_filenames)}")
                        # Check for potential encoding/whitespace issues
                        for doc in available_original_filenames:
                            if doc and document_name in doc:
                                logger.warning(f"FILTERING: Partial match found: '{doc}' contains '{document_name}'")
                else:
                    search_results = all_results[:limit]
                
                # Ensure we always return exactly k=8 results (or limit specified)
                search_results = search_results[:limit]
                    
                logger.info(f"DATABASE: SQL search returned {len(search_results)} results")
                
            except Exception as sql_error:
                logger.warning(f"SQL function failed: {str(sql_error)}")
                logger.info("DATABASE: Falling back to manual filtering...")
                
                # Fallback: Use regular search and filter by document name
                fallback_result = self.supabase.rpc('search_similar_embeddings', {
                    'query_embedding': embedding_list,
                    'similarity_threshold': similarity_threshold,
                    'match_count': limit * 2  # Get more results to filter
                }).execute()
                
                all_results = fallback_result.data if fallback_result.data else []
                
                # Filter by document name manually (use original_filename field)
                if document_name:
                    logger.info(f"FALLBACK FILTERING: Target document name: '{document_name}'")
                    logger.info(f"FALLBACK FILTERING: Before filtering - {len(all_results)} total results")
                    
                    # Debug: Log all available document names in fallback from both fields
                    fallback_docs = set()
                    fallback_original_filenames = set()
                    for r in all_results[:5]:  # Log first 5 for debugging
                        doc_name = r.get('document_name', 'N/A')
                        original_filename = r.get('original_filename', 'N/A')
                        fallback_docs.add(doc_name)
                        fallback_original_filenames.add(original_filename)
                        logger.info(f"FALLBACK FILTERING: Result has document_name: '{doc_name}', original_filename: '{original_filename}'")
                    
                    # Use both original_filename and filename for filtering
                    search_results = [r for r in all_results if 
                                    r.get('original_filename') == document_name or r.get('filename') == document_name][:limit]
                    logger.info(f"FALLBACK FILTERING: After filtering by original_filename/filename - {len(search_results)} matching results")
                    
                    if len(search_results) == 0:
                        logger.warning(f"FALLBACK FILTERING: NO MATCHES found for document_name='{document_name}'")
                        logger.warning(f"FALLBACK FILTERING: Available original_filenames: {sorted(fallback_original_filenames)}")
                else:
                    search_results = all_results[:limit]
                    logger.info(f"DATABASE: Fallback returned {len(search_results)} results")
            
            # Log first few results for debugging
            for i, res in enumerate(search_results[:2]):
                logger.info(f"RESULT {i+1}: doc_name='{res.get('document_name', 'N/A')}', similarity={res.get('similarity_score', 0):.3f}")
                logger.info(f"RESULT {i+1}: text='{res.get('chunk_text', '')[:100]}...'")
            
            logger.info(f"=== EMBEDDING SEARCH END: {len(search_results)} results ===")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in document name-filtered search: {str(e)}")
            return []
    
    def search_similar_documents_by_id(self, query: str, document_id: str, 
                                     limit: int = 5, similarity_threshold: float = 0.5):
        """Search similar documents filtered by exact document_id match"""
        try:
            logger.info(f"=== DOCUMENT_ID SEARCH START ===")
            logger.info(f"SEARCH PARAMS: query='{query[:50]}...', limit={limit}")
            logger.info(f"SEARCH PARAMS: threshold={similarity_threshold}, document_id='{document_id}'")
            
            # Generate query embedding
            logger.info("ENCODING: Generating query embedding...")
            query_embedding = self.model.encode([query])[0].tolist()
            logger.info(f"ENCODING: Query encoded to {len(query_embedding)} dimensions")
            
            # Search with exact document_id filter
            logger.info(f"DATABASE: Searching embeddings for document_id='{document_id}'")
            
            try:
                # Include 'embedding' column to compute actual similarity
                response = self.supabase.table('document_embeddings')\
                    .select('id, document_id, chunk_id, chunk_text, chunk_index, document_name, metadata, embedding')\
                    .eq('document_id', document_id)\
                    .execute()
                
                if not response.data:
                    logger.warning(f"FILTERING: NO EMBEDDINGS found for document_id='{document_id}'")
                    return []
                
                logger.info(f"FILTERING: Found {len(response.data)} embeddings for document_id='{document_id}'")
                
                # Calculate similarities for filtered embeddings
                import numpy as np
                results = []
                for row in response.data:
                    try:
                        # Get stored embedding and compute actual similarity
                        if 'embedding' not in row or not row['embedding']:
                            continue
                        
                        stored_embedding = row['embedding']
                        
                        # Parse embedding if it's stored as a JSON string
                        if isinstance(stored_embedding, str):
                            try:
                                stored_embedding = json.loads(stored_embedding)
                            except json.JSONDecodeError:
                                continue
                        
                        # Validate embedding is a list/array
                        if not isinstance(stored_embedding, (list, tuple)):
                            continue
                            
                        if len(stored_embedding) == 0:
                            continue
                        
                        # Calculate actual cosine similarity
                        similarity = self._cosine_similarity(query_embedding, stored_embedding)
                        
                        # Only include if similarity meets threshold
                        if similarity >= similarity_threshold:
                            results.append({
                                'document_id': row['document_id'],
                                'chunk_id': row['chunk_id'],
                                'chunk_text': row['chunk_text'],
                                'chunk_index': row['chunk_index'],
                                'similarity_score': similarity,
                                'original_filename': row.get('document_name', 'Unknown'),
                                'file_type': 'pdf',  # Default since most are PDFs
                                'metadata': row.get('metadata', {})
                            })
                        
                    except Exception as row_error:
                        logger.warning(f"Error processing row: {str(row_error)}")
                        continue
                
                # Sort by similarity score (highest first), then by chunk_index
                results.sort(key=lambda x: (-x.get('similarity_score', 0), x.get('chunk_index', 0)))
                
                # Limit results
                results = results[:limit]
                
                logger.info(f"SEARCH COMPLETE: Returning {len(results)} chunks for document_id='{document_id}' with similarity >= {similarity_threshold}")
                for i, result in enumerate(results[:3]):
                    logger.info(f"TOP RESULT {i+1}: similarity={result['similarity_score']:.3f}, text='{result['chunk_text'][:50]}...'")
                
                return results
                
            except Exception as search_error:
                logger.error(f"Database search error for document_id {document_id}: {str(search_error)}")
                return []
            
        except Exception as e:
            logger.error(f"Error in search_similar_documents_by_id: {str(e)}")
            return []
    
    # ========== MULTIMODAL CLIP FUNCTIONALITY ==========
    
    async def create_multimodal_embeddings(self, document_id: str, text: str, images_data: List[Dict], document_name: str = None) -> Dict:
        """
        Create embeddings for both text and images using CLIP
        
        Args:
            document_id: Document identifier
            text: Extracted text content
            images_data: List of image data with CLIP embeddings
            document_name: Optional document name
            
        Returns:
            Dictionary with embedding creation results
        """
        try:
            from services.clip_service import get_clip_service
            
            logger.info(f"Creating multimodal embeddings for document {document_id}")
            
            # Get document name if not provided
            if document_name is None:
                document_name = self._get_document_name(document_id)
            
            # Create text embeddings (existing functionality)
            text_embeddings_count = await self.create_embeddings(document_id, text, document_name)
            
            # Create image embeddings (new CLIP functionality)
            image_embeddings_count = 0
            
            if images_data and len(images_data) > 0:
                logger.info(f"Creating CLIP embeddings for {len(images_data)} images")
                
                batch_data = []
                
                for img_data in images_data:
                    try:
                        if not img_data.get('clip_embedding'):
                            logger.warning(f"Image {img_data.get('image_index', 'unknown')} missing CLIP embedding")
                            continue
                        
                        # Handle CLIP embedding dimension (512) vs text embedding dimension (768)
                        clip_embedding = img_data['clip_embedding']
                        
                        # Pad CLIP embedding to match text embedding dimensions (768)
                        if len(clip_embedding) == 512:
                            # Pad with zeros to reach 768 dimensions
                            padded_embedding = clip_embedding + [0.0] * (768 - 512)
                        else:
                            padded_embedding = clip_embedding
                        
                        # Prepare image embedding data for Supabase
                        image_embedding_data = {
                            'document_id': document_id,
                            'document_name': document_name,
                            'chunk_id': f"{document_id}_image_{img_data['image_index']}",
                            'chunk_text': f"Page {img_data['page_num']} - {img_data.get('description', 'Technical drawing')}",
                            'chunk_index': img_data['image_index'],
                            'embedding': padded_embedding,  # Use padded CLIP embedding
                            'metadata': {
                                'type': 'image',
                                'source_type': img_data.get('source_type', 'rendered_page'),
                                'page_num': img_data['page_num'],
                                'width': img_data['width'],
                                'height': img_data['height'],
                                'description': img_data.get('description', ''),
                                'clip_processed': True,
                                'original_clip_dims': len(clip_embedding),
                                'padded_dims': len(padded_embedding)
                            }
                        }
                        
                        batch_data.append(image_embedding_data)
                        image_embeddings_count += 1
                        
                    except Exception as img_error:
                        logger.warning(f"Failed to prepare image embedding {img_data.get('image_index', 'unknown')}: {str(img_error)}")
                        continue
                
                # Insert image embeddings in batch
                if batch_data:
                    try:
                        response = self.supabase.table("document_embeddings").insert(batch_data).execute()
                        logger.info(f"Successfully stored {len(batch_data)} image embeddings")
                    except Exception as insert_error:
                        logger.error(f"Failed to insert image embeddings: {str(insert_error)}")
                        image_embeddings_count = 0
            
            return {
                'success': True,
                'text_embeddings': text_embeddings_count,
                'image_embeddings': image_embeddings_count,
                'total_embeddings': text_embeddings_count + image_embeddings_count,
                'message': f'Created {text_embeddings_count} text + {image_embeddings_count} image embeddings'
            }
            
        except Exception as e:
            logger.error(f"Error creating multimodal embeddings: {str(e)}")
            return {
                'success': False,
                'text_embeddings': 0,
                'image_embeddings': 0,
                'total_embeddings': 0,
                'error': str(e),
                'message': 'Multimodal embedding creation failed'
            }
    
    async def search_multimodal_context(self, query: str, max_chunks: int = 8, 
                                       include_images: bool = True, similarity_threshold: float = 0.3,
                                       document_id: str = None) -> Dict:
        """
        Search both text and image content for multimodal context
        
        Args:
            query: Search query
            max_chunks: Maximum chunks to return
            include_images: Whether to include image results
            similarity_threshold: Minimum similarity threshold
            document_id: Optional document ID to filter results
            
        Returns:
            Dictionary with multimodal search results
        """
        try:
            search_start_time = time.time()
            logger.info(f"🔍 MULTIMODAL SEARCH STARTED")
            logger.info(f"   Query: '{query[:100]}{'...' if len(query) > 100 else ''}'") 
            logger.info(f"   Max chunks: {max_chunks}, Include images: {include_images}")
            logger.info(f"   Similarity threshold: {similarity_threshold}, Document filter: {document_id or 'None'}")
            
            # Search text embeddings with optional document filtering
            # For technical drawings, prioritize text results as they contain semantic information
            text_limit = max_chunks if not include_images else max(max_chunks // 2, 3)  # Ensure at least 3 text results
            
            # Search text embeddings with robust error handling
            text_results = None
            text_search_error = None
            
            try:
                if document_id:
                    logger.info(f"📄 TEXT SEARCH: Filtering by document_id '{document_id}' (limit: {text_limit})")
                    text_results = self.search_similar_documents_by_id(
                        query=query,
                        document_id=document_id,
                        limit=text_limit,
                        similarity_threshold=similarity_threshold
                    )
                else:
                    logger.info(f"📄 TEXT SEARCH: All documents (limit: {text_limit})")
                    text_results = self.search_similar_documents(
                        query=query,
                        limit=text_limit,
                        similarity_threshold=similarity_threshold
                    )
                
                # Log text search results
                if hasattr(text_results, 'results'):
                    result_count = len(text_results.results)
                elif isinstance(text_results, list):
                    result_count = len(text_results)
                else:
                    result_count = 0
                logger.info(f"✅ TEXT SEARCH: Found {result_count} results")
                    
            except Exception as text_error:
                text_search_error = str(text_error)
                logger.error(f"❌ TEXT SEARCH FAILED: {text_search_error}")
                text_results = []  # Fallback to empty results
            
            image_results = []
            
            # Search image embeddings if requested with enhanced error handling
            image_search_error = None
            if include_images:
                try:
                    logger.info(f"🖼️  IMAGE SEARCH: Starting CLIP-based search (limit: {max_chunks // 2 if include_images else max_chunks})")
                    from services.clip_service import get_clip_service
                    clip_service = get_clip_service()
                    
                    if not clip_service:
                        logger.warning("⚠️  CLIP service not available, skipping image search")
                        image_search_error = "CLIP service not available"
                    else:
                        logger.info("✅ CLIP service initialized successfully")
                        # Get CLIP text embedding for the query
                        query_embedding = clip_service.get_text_embedding(query)
                        
                        if query_embedding:
                            # Pad CLIP query embedding to match database dimensions (768)
                            if len(query_embedding) == 512:
                                padded_query_embedding = query_embedding + [0.0] * (768 - 512)
                            else:
                                padded_query_embedding = query_embedding
                            
                            # Search for similar image embeddings with optional document filtering
                            query_builder = self.supabase.table("document_embeddings").select(
                                "document_id, document_name, chunk_id, chunk_text, chunk_index, metadata, embedding"
                            )
                            
                            # Apply document filtering if document_id is provided
                            if document_id:
                                query_builder = query_builder.eq('document_id', document_id)
                                
                            embeddings_response = query_builder.execute()
                            
                            if embeddings_response.data:
                                import numpy as np
                                
                                image_similarities = []
                                
                                for row in embeddings_response.data:
                                    try:
                                        # Check if this is an image embedding
                                        metadata = row.get('metadata', {})
                                        if not isinstance(metadata, dict) or metadata.get('type') != 'image':
                                            continue
                                        
                                        stored_embedding = row['embedding']
                                        if stored_embedding:
                                            # Parse embedding if it's stored as a JSON string
                                            if isinstance(stored_embedding, str):
                                                try:
                                                    stored_embedding = json.loads(stored_embedding)
                                                except json.JSONDecodeError:
                                                    logger.warning(f"Could not parse image embedding for {row['chunk_id']}")
                                                    continue
                                            
                                            # Validate embedding is a list
                                            if not isinstance(stored_embedding, (list, tuple)) or len(stored_embedding) == 0:
                                                continue
                                            
                                            # For image embeddings, only compare the first 512 dimensions (original CLIP)
                                            stored_clip_part = stored_embedding[:512]
                                            query_clip_part = query_embedding[:512]  # Use original 512-dim query
                                            
                                            # Calculate similarity using CLIP dimensions only
                                            similarity = self._cosine_similarity(query_clip_part, stored_clip_part)
                                            
                                            if similarity >= similarity_threshold:
                                                image_similarities.append({
                                                    'document_id': row['document_id'],
                                                    'document_name': row['document_name'],
                                                    'chunk_id': row['chunk_id'],
                                                    'chunk_text': row['chunk_text'],
                                                    'chunk_index': row['chunk_index'],
                                                    'similarity_score': similarity,
                                                    'metadata': metadata,
                                                    'type': 'image'
                                                })
                                    
                                    except Exception as img_search_error:
                                        logger.warning(f"Error processing image embedding: {str(img_search_error)}")
                                        continue
                                
                                # Sort by similarity and take top results
                                image_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                                image_results = image_similarities[:max_chunks // 2 if include_images else max_chunks]
                                
                                logger.info(f"✅ IMAGE SEARCH: Found {len(image_results)} relevant results (threshold: {similarity_threshold})")
                                if image_results:
                                    avg_score = sum(r['similarity_score'] for r in image_results) / len(image_results)
                                    max_score = max(r['similarity_score'] for r in image_results)
                                    logger.info(f"   Image scores - Average: {avg_score:.3f}, Max: {max_score:.3f}")
                
                except ImportError as import_error:
                    image_search_error = f"CLIP service import failed: {str(import_error)}"
                    logger.warning(f"Could not import CLIP service: {image_search_error}")
                except Exception as clip_error:
                    image_search_error = f"CLIP image search failed: {str(clip_error)}"
                    logger.warning(f"CLIP image search failed: {image_search_error}")
            else:
                logger.info("🚫 IMAGE SEARCH: Disabled (include_images=False)")
            
            # Combine and rank results
            all_results = []
            
            # Process text results with enhanced format handling
            text_result_list = []
            text_processing_errors = []
            
            try:
                if text_results is None:
                    logger.warning("Text results is None, using empty list")
                elif hasattr(text_results, 'results'):
                    # SimilaritySearchResponse format
                    text_result_list = text_results.results or []
                    logger.info(f"Processing {len(text_result_list)} text results from SimilaritySearchResponse")
                elif isinstance(text_results, list):
                    # Direct list format
                    text_result_list = text_results
                    logger.info(f"Processing {len(text_result_list)} text results from list")
                elif hasattr(text_results, '__iter__'):
                    # Try to convert iterable to list
                    text_result_list = list(text_results)
                    logger.info(f"Converted iterable to {len(text_result_list)} text results")
                else:
                    logger.warning(f"Unexpected text results format: {type(text_results)}, using empty list")
                    text_processing_errors.append(f"Unexpected format: {type(text_results)}")
                    
            except Exception as format_error:
                logger.error(f"Error processing text results format: {str(format_error)}")
                text_processing_errors.append(str(format_error))
                text_result_list = []
            
            # Process individual text results with robust error handling
            processed_text_count = 0
            skipped_text_count = 0
            
            for i, result in enumerate(text_result_list):
                try:
                    result_dict = None
                    
                    if hasattr(result, 'document_id'):
                        # Convert result object to dict by extracting attributes
                        result_dict = {
                            'document_id': getattr(result, 'document_id', 'unknown'),
                            'chunk_id': getattr(result, 'chunk_id', 'unknown'),
                            'chunk_text': getattr(result, 'chunk_text', ''),
                            'similarity_score': getattr(result, 'similarity_score', 0),
                            'original_filename': getattr(result, 'original_filename', 'unknown'),
                            'document_name': getattr(result, 'original_filename', 'unknown'),  # Alias
                            'chunk_index': getattr(result, 'chunk_index', None),
                            'metadata': getattr(result, 'metadata', {})
                        }
                    elif isinstance(result, dict):
                        # Ensure required fields exist with defaults
                        result_dict = {
                            'document_id': result.get('document_id', 'unknown'),
                            'chunk_id': result.get('chunk_id', f'chunk_{i}'),
                            'chunk_text': result.get('chunk_text', ''),
                            'similarity_score': result.get('similarity_score', 0),
                            'original_filename': result.get('original_filename', result.get('document_name', 'unknown')),
                            'document_name': result.get('document_name', result.get('original_filename', 'unknown')),
                            'chunk_index': result.get('chunk_index', i),
                            'metadata': result.get('metadata', {})
                        }
                    else:
                        logger.warning(f"Skipping text result {i}: unexpected type {type(result)}")
                        skipped_text_count += 1
                        continue
                    
                    # Validate essential fields
                    if not result_dict.get('chunk_text', '').strip():
                        logger.warning(f"Skipping text result {i}: empty chunk_text")
                        skipped_text_count += 1
                        continue
                    
                    # Apply document type-specific optimizations
                    base_score = max(0, min(1, result_dict.get('similarity_score', 0)))  # Clamp between 0 and 1
                    
                    # Determine document type from metadata or filename
                    doc_metadata = result_dict.get('metadata', {})
                    original_filename = result_dict.get('original_filename', '').lower()
                    
                    # Office document type detection and optimization
                    doc_type = 'unknown'
                    if original_filename:
                        if any(ext in original_filename for ext in ['.xlsx', '.xls']):
                            doc_type = 'excel'
                        elif any(ext in original_filename for ext in ['.pptx', '.ppt']):
                            doc_type = 'powerpoint'
                        elif any(ext in original_filename for ext in ['.docx', '.doc']):
                            doc_type = 'word'
                        elif '.pdf' in original_filename:
                            doc_type = 'pdf'
                    
                    # Apply type-specific scoring boosts
                    text_boost = 1.0
                    chunk_text = result_dict.get('chunk_text', '').lower()
                    
                    if doc_type == 'excel':
                        # Excel documents: boost results with structured data patterns
                        if any(indicator in chunk_text for indicator in ['===', '\t', 'sheet:', 'row', 'column']):
                            text_boost = 1.3  # Strong boost for tabular data
                        elif base_score > 0.3:
                            text_boost = 1.15  # Moderate boost for relevant Excel content
                    elif doc_type == 'powerpoint':
                        # PowerPoint: boost slide titles and key content
                        if any(indicator in chunk_text for indicator in ['slide', 'title:', 'bullet', '•']):
                            text_boost = 1.25  # Boost slide structure content
                        elif base_score > 0.35:
                            text_boost = 1.1  # Moderate boost for PPT content
                    elif doc_type == 'word':
                        # Word documents: boost headers and structured content
                        if any(indicator in chunk_text for indicator in ['heading', 'title', 'section']):
                            text_boost = 1.2  # Boost structured Word content
                        elif base_score > 0.3:
                            text_boost = 1.1
                    elif doc_type == 'pdf':
                        # PDF: standard boost for high-quality matches (original behavior)
                        text_boost = 1.2 if base_score > 0.4 else 1.0
                    else:
                        # Unknown/other formats: conservative boost
                        text_boost = 1.1 if base_score > 0.4 else 1.0
                    
                    # Add document type to result metadata
                    result_dict['detected_type'] = doc_type
                    
                    all_results.append({
                        **result_dict,
                        'type': 'text',
                        'multimodal_score': base_score * text_boost,
                        'document_type': doc_type,
                        'score_boost': text_boost
                    })
                    processed_text_count += 1
                    
                except Exception as text_error:
                    logger.warning(f"Error processing text result {i}: {str(text_error)}")
                    text_processing_errors.append(f"Result {i}: {str(text_error)}")
                    skipped_text_count += 1
                    continue
                    
            logger.info(f"✅ TEXT PROCESSING: {processed_text_count} processed, {skipped_text_count} skipped")
            
            # Log document type distribution for text results
            if processed_text_count > 0:
                doc_type_counts = {}
                score_stats = {'min': 1.0, 'max': 0.0, 'sum': 0.0, 'count': 0}
                for result in all_results:
                    if result.get('type') == 'text':
                        doc_type = result.get('document_type', 'unknown')
                        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                        score = result.get('multimodal_score', 0)
                        score_stats['min'] = min(score_stats['min'], score)
                        score_stats['max'] = max(score_stats['max'], score)
                        score_stats['sum'] += score
                        score_stats['count'] += 1
                
                type_summary = [f"{t}:{c}" for t, c in doc_type_counts.items()]
                avg_score = score_stats['sum'] / score_stats['count'] if score_stats['count'] > 0 else 0
                logger.info(f"   Text types: {', '.join(type_summary)}")
                logger.info(f"   Text scores - Min: {score_stats['min']:.3f}, Avg: {avg_score:.3f}, Max: {score_stats['max']:.3f}")
            
            # Add image results with office document optimizations
            for result in image_results:
                try:
                    # Apply document type-specific image scoring
                    original_filename = result.get('document_name', '').lower()
                    base_score = result['similarity_score']
                    
                    # Determine document type for image content
                    doc_type = 'unknown'
                    if original_filename:
                        if any(ext in original_filename for ext in ['.xlsx', '.xls']):
                            doc_type = 'excel'
                        elif any(ext in original_filename for ext in ['.pptx', '.ppt']):
                            doc_type = 'powerpoint'
                        elif any(ext in original_filename for ext in ['.docx', '.doc']):
                            doc_type = 'word'
                        elif '.pdf' in original_filename:
                            doc_type = 'pdf'
                    
                    # Office documents often have more relevant visual content
                    image_boost = 1.1  # Default boost
                    if doc_type == 'powerpoint':
                        # PowerPoint images are often central to content
                        image_boost = 1.4 if base_score > 0.3 else 1.2
                    elif doc_type == 'excel':
                        # Excel charts and graphs are highly relevant
                        image_boost = 1.3 if base_score > 0.25 else 1.15
                    elif doc_type == 'word':
                        # Word document images/diagrams are important
                        image_boost = 1.25 if base_score > 0.3 else 1.1
                    elif doc_type == 'pdf':
                        # PDF images can be very relevant for technical content
                        image_boost = 1.2 if base_score > 0.3 else 1.1
                    
                    all_results.append({
                        **result,
                        'type': 'image',
                        'document_type': doc_type,
                        'multimodal_score': base_score * image_boost,
                        'score_boost': image_boost
                    })
                    
                except Exception as img_proc_error:
                    logger.warning(f"⚠️  Error processing image result: {str(img_proc_error)}")
                    # Fallback to basic processing
                    all_results.append({
                        **result,
                        'type': 'image',
                        'document_type': 'unknown',
                        'multimodal_score': result['similarity_score'] * 1.1,
                        'score_boost': 1.1
                    })
            
            # Log pre-sorting statistics
            total_results = len(all_results)
            text_count_pre = len([r for r in all_results if r.get('type') == 'text'])
            image_count_pre = len([r for r in all_results if r.get('type') == 'image'])
            logger.info(f"🔀 RESULT RANKING: Sorting {total_results} results ({text_count_pre} text, {image_count_pre} image)")
            
            if all_results:
                scores_pre = [r.get('multimodal_score', 0) for r in all_results]
                logger.info(f"   Score range: {min(scores_pre):.3f} - {max(scores_pre):.3f}")
            
            # Sort by multimodal score and limit results
            all_results.sort(key=lambda x: x.get('multimodal_score', 0), reverse=True)
            final_results = all_results[:max_chunks]
            
            # Log final selection
            logger.info(f"🏆 FINAL SELECTION: Top {len(final_results)} of {total_results} results")
            if final_results:
                final_scores = [r.get('multimodal_score', 0) for r in final_results]
                logger.info(f"   Final score range: {min(final_scores):.3f} - {max(final_scores):.3f}")
                
                # Log document type distribution in final results
                final_doc_types = {}
                for result in final_results:
                    doc_type = result.get('document_type', 'unknown')
                    result_type = result.get('type', 'unknown')
                    key = f"{doc_type}_{result_type}"
                    final_doc_types[key] = final_doc_types.get(key, 0) + 1
                
                type_summary = [f"{k}:{v}" for k, v in final_doc_types.items()]
                logger.info(f"   Final distribution: {', '.join(type_summary)}")
            
            # Build comprehensive response with diagnostic information
            text_count = len([r for r in final_results if r.get('type') == 'text'])
            image_count = len([r for r in final_results if r.get('type') == 'image'])
            
            # Create detailed status message
            status_parts = []
            if text_search_error:
                status_parts.append(f"text search failed: {text_search_error}")
            else:
                status_parts.append(f"{text_count} text results")
                
            if image_search_error:
                status_parts.append(f"image search failed: {image_search_error}")
            elif include_images:
                status_parts.append(f"{image_count} image results")
            else:
                status_parts.append("images disabled")
                
            if text_processing_errors:
                status_parts.append(f"{len(text_processing_errors)} processing errors")
            
            status_message = f"Multimodal search completed: {', '.join(status_parts)}"
            
            response = {
                'success': True,
                'results': final_results,
                'total_results': len(final_results),
                'text_results': text_count,
                'image_results': image_count,
                'query': query,
                'document_id': document_id,
                'similarity_threshold': similarity_threshold,
                'message': status_message
            }
            
            # Add diagnostic information if there were issues
            if text_search_error or image_search_error or text_processing_errors:
                response['diagnostics'] = {
                    'text_search_error': text_search_error,
                    'image_search_error': image_search_error, 
                    'text_processing_errors': text_processing_errors[:5]  # Limit to first 5 errors
                }
                
            # Log final performance metrics
            search_duration = time.time() - search_start_time
            logger.info(f"⏱️  SEARCH COMPLETED: {search_duration:.2f}s total")
            logger.info(f"📊 FINAL METRICS: {status_message}")
            
            # Log performance breakdown
            if final_results:
                content_chars = sum(len(str(r.get('chunk_text', ''))) for r in final_results if r.get('type') == 'text')
                logger.info(f"   Content volume: {content_chars:,} characters in text chunks")
                
            return response
            
        except Exception as e:
            logger.error(f"Critical error in multimodal search: {str(e)}", exc_info=True)
            return {
                'success': False,
                'results': [],
                'total_results': 0,
                'text_results': 0,
                'image_results': 0,
                'query': query,
                'document_id': document_id,
                'similarity_threshold': similarity_threshold,
                'error': str(e),
                'message': f'Multimodal search failed: {str(e)}',
                'diagnostics': {
                    'critical_error': str(e),
                    'error_type': type(e).__name__
                }
            }