import clip
import torch
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import io
import base64
import logging
from typing import List, Dict, Optional, Tuple
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

class CLIPService:
    """Lightweight CLIP service for multimodal document processing"""
    
    def __init__(self):
        """Initialize CLIP model"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Configuration
            self.image_embedding_dim = 512  # CLIP ViT-B/32 output dimension
            self.max_images_per_doc = 20
            self.min_image_size = (32, 32)  # Minimum image size to process
            self.extraction_dpi = 150
            
            # Thread pool for CPU-intensive operations
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            logger.info(f"CLIP service initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP service: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str, max_images: int = None) -> List[Dict]:
        """
        Extract images from PDF with metadata
        For CAD/vector PDFs, renders pages as images for CLIP processing
        
        Args:
            pdf_path: Path to PDF file
            max_images: Maximum number of images to extract
            
        Returns:
            List of image dictionaries with metadata
        """
        if max_images is None:
            max_images = self.max_images_per_doc
            
        images_data = []
        
        try:
            # Open PDF document
            pdf_doc = fitz.open(pdf_path)
            logger.info(f"Extracting content from PDF: {pdf_path} ({pdf_doc.page_count} pages)")
            
            image_count = 0
            
            # First, try to extract embedded bitmap images
            embedded_images = 0
            for page_num in range(pdf_doc.page_count):
                if image_count >= max_images:
                    break
                    
                page = pdf_doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    if image_count >= max_images:
                        break
                    
                    try:
                        # Extract embedded image
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_doc, xref)
                        
                        # Skip if image is too small
                        if pix.width < self.min_image_size[0] or pix.height < self.min_image_size[1]:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            pix1 = None
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        img_pil.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        image_metadata = {
                            'image_index': image_count,
                            'page_num': page_num + 1,
                            'source_type': 'embedded_image',
                            'pdf_image_index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'image_base64': img_base64,
                            'size_bytes': len(img_base64)
                        }
                        
                        images_data.append(image_metadata)
                        image_count += 1
                        embedded_images += 1
                        
                        pix = None
                        
                    except Exception as img_error:
                        logger.warning(f"Failed to extract embedded image {img_index} from page {page_num + 1}: {str(img_error)}")
                        continue
            
            logger.info(f"Extracted {embedded_images} embedded images")
            
            # If no embedded images found, render pages as images (for CAD/vector PDFs)
            if embedded_images == 0 and image_count < max_images:
                logger.info("No embedded images found. Rendering PDF pages as images for CLIP processing...")
                
                # Render each page as an image
                for page_num in range(min(pdf_doc.page_count, max_images - image_count)):
                    try:
                        page = pdf_doc[page_num]
                        
                        # Check if page has visual content (drawings)
                        drawings = page.get_drawings()
                        if len(drawings) == 0:
                            continue  # Skip text-only pages
                        
                        # Render page as image with high quality
                        # Use 2x scale for better quality
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        img_pil.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        # Create page image metadata
                        page_metadata = {
                            'image_index': image_count,
                            'page_num': page_num + 1,
                            'source_type': 'rendered_page',
                            'pdf_image_index': -1,  # Not an embedded image
                            'width': pix.width,
                            'height': pix.height,
                            'image_base64': img_base64,
                            'size_bytes': len(img_base64),
                            'drawing_objects': len(drawings)
                        }
                        
                        images_data.append(page_metadata)
                        image_count += 1
                        
                        pix = None
                        logger.info(f"Rendered page {page_num + 1} as image ({len(drawings)} drawing objects)")
                        
                    except Exception as page_error:
                        logger.warning(f"Failed to render page {page_num + 1} as image: {str(page_error)}")
                        continue
            
            pdf_doc.close()
            logger.info(f"Successfully processed {len(images_data)} images/pages from PDF")
            
            return images_data
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF {pdf_path}: {str(e)}")
            return []
    
    def analyze_image_with_clip(self, image: Image.Image) -> Dict:
        """
        Analyze single image with CLIP
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with CLIP embedding and metadata
        """
        try:
            # Preprocess image for CLIP
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate CLIP embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features.cpu().numpy().flatten()
            
            # Normalize embedding
            image_features = image_features / np.linalg.norm(image_features)
            
            return {
                'clip_embedding': image_features.tolist(),
                'embedding_dim': len(image_features),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image with CLIP: {str(e)}")
            return {
                'clip_embedding': None,
                'embedding_dim': 0,
                'success': False,
                'error': str(e)
            }
    
    def generate_image_description(self, image: Image.Image, context_text: str = "") -> str:
        """
        Generate text description of image using CLIP text similarity
        
        Args:
            image: PIL Image object
            context_text: Additional context from document
            
        Returns:
            Text description of image content
        """
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Candidate descriptions for engineering drawings
            descriptions = [
                "a technical drawing",
                "an architectural floor plan", 
                "a construction diagram",
                "an engineering schematic",
                "a building layout",
                "room dimensions and layout",
                "construction specifications",
                "structural details",
                "electrical wiring diagram",
                "plumbing layout",
                "mechanical drawing"
            ]
            
            # Add context-specific descriptions
            if "room" in context_text.lower():
                descriptions.extend(["room layout", "floor plan with rooms", "residential layout"])
            if "dimension" in context_text.lower():
                descriptions.extend(["dimensional drawing", "measured layout", "scaled drawing"])
            
            # Tokenize descriptions
            text_inputs = clip.tokenize(descriptions).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).squeeze(0)
                best_match_idx = similarities.argmax().item()
                confidence = similarities[best_match_idx].item()
            
            best_description = descriptions[best_match_idx]
            
            # Add confidence-based prefix
            if confidence > 0.25:
                return f"{best_description}"
            else:
                return "technical document image"
                
        except Exception as e:
            logger.warning(f"Error generating image description: {str(e)}")
            return "document image"
    
    async def process_document_images(self, pdf_path: str, extracted_text: str = "", max_images: int = None) -> Dict:
        """
        Process all images in document with CLIP analysis
        
        Args:
            pdf_path: Path to PDF file
            extracted_text: Extracted text for context
            max_images: Maximum images to process
            
        Returns:
            Dictionary with processed images and embeddings
        """
        try:
            logger.info(f"Starting CLIP processing for document: {pdf_path}")
            
            # Extract images from PDF
            loop = asyncio.get_event_loop()
            images_data = await loop.run_in_executor(
                self.executor, 
                self.extract_images_from_pdf, 
                pdf_path, 
                max_images
            )
            
            if not images_data:
                return {
                    'success': True,
                    'images': [],
                    'num_images': 0,
                    'message': 'No images found in document'
                }
            
            processed_images = []
            
            # Process each image with CLIP
            for img_data in images_data:
                try:
                    # Decode base64 image
                    img_bytes = base64.b64decode(img_data['image_base64'])
                    img_pil = Image.open(io.BytesIO(img_bytes))
                    
                    # Analyze with CLIP
                    clip_analysis = self.analyze_image_with_clip(img_pil)
                    
                    # Generate description
                    description = self.generate_image_description(img_pil, extracted_text)
                    
                    # Combine metadata
                    processed_image = {
                        **img_data,
                        'clip_embedding': clip_analysis['clip_embedding'],
                        'description': description,
                        'clip_success': clip_analysis['success']
                    }
                    
                    processed_images.append(processed_image)
                    
                except Exception as img_error:
                    logger.warning(f"Failed to process image {img_data.get('image_index', 'unknown')}: {str(img_error)}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_images)} images with CLIP")
            
            return {
                'success': True,
                'images': processed_images,
                'num_images': len(processed_images),
                'message': f'Processed {len(processed_images)} images with CLIP analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in CLIP document processing: {str(e)}")
            return {
                'success': False,
                'images': [],
                'num_images': 0,
                'error': str(e),
                'message': 'CLIP processing failed'
            }
    
    def search_similar_images(self, query_text: str, image_embeddings: List[List[float]], image_descriptions: List[str], top_k: int = 5) -> List[Dict]:
        """
        Search for images similar to text query using CLIP
        
        Args:
            query_text: Text query
            image_embeddings: List of image embeddings
            image_descriptions: List of image descriptions
            top_k: Number of top results to return
            
        Returns:
            List of similar images with scores
        """
        try:
            if not image_embeddings:
                return []
            
            # Encode query text
            text_input = clip.tokenize([query_text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = text_features.cpu().numpy().flatten()
                text_features = text_features / np.linalg.norm(text_features)
            
            # Calculate similarities
            similarities = []
            for i, img_embedding in enumerate(image_embeddings):
                if img_embedding:  # Check if embedding exists
                    img_emb = np.array(img_embedding)
                    similarity = np.dot(text_features, img_emb)
                    similarities.append({
                        'index': i,
                        'similarity': float(similarity),
                        'description': image_descriptions[i] if i < len(image_descriptions) else "No description"
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in image similarity search: {str(e)}")
            return []
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate CLIP text embedding for comparison with images
        
        Args:
            text: Input text
            
        Returns:
            CLIP text embedding as list
        """
        try:
            text_input = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = text_features.cpu().numpy().flatten()
                text_features = text_features / np.linalg.norm(text_features)
            
            return text_features.tolist()
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            return None
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Global instance
_clip_service_instance = None

def get_clip_service() -> Optional[CLIPService]:
    """
    Get or create a singleton instance of the CLIP service
    
    Returns:
        CLIPService instance or None if initialization fails
    """
    global _clip_service_instance
    
    if _clip_service_instance is None:
        try:
            _clip_service_instance = CLIPService()
            logger.info("CLIP service singleton created")
        except Exception as e:
            logger.error(f"Failed to create CLIP service: {str(e)}")
            return None
    
    return _clip_service_instance