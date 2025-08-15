import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import logging
from typing import Optional, List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class OCRService:
    """OCR service using Tesseract for text extraction"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # OCR configuration
        self.tesseract_config = {
            'lang': 'eng',
            'config': '--psm 3 --oem 3',  # Page segmentation mode 3, OCR Engine Mode 3
        }
        
        # Configure Tesseract path for conda environment
        pytesseract.pytesseract.tesseract_cmd = r'/mnt/data/miniconda3/envs/py311/bin/tesseract'
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF and OCR fallback
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text string
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._extract_text_from_pdf_sync,
                pdf_path
            )
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return f"Error extracting text: {str(e)}"
    
    def _extract_text_from_pdf_sync(self, pdf_path: str) -> str:
        """Synchronous PDF text extraction"""
        extracted_text = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                # First, try to extract text directly
                direct_text = page.get_text().strip()
                
                if direct_text:
                    # Text found directly, use it
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{direct_text}")
                    logger.info(f"Direct text extraction successful for page {page_num + 1}")
                else:
                    # No direct text, use OCR
                    logger.info(f"Using OCR for page {page_num + 1}")
                    ocr_text = self._ocr_page(page, page_num + 1)
                    
                    if ocr_text.strip():
                        extracted_text.append(f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}")
                    else:
                        extracted_text.append(f"--- Page {page_num + 1} ---\n[No text detected]")
            
            pdf_doc.close()
            
            # Combine all extracted text
            full_text = '\n\n'.join(extracted_text)
            
            if not full_text.strip():
                return "No text could be extracted from this document."
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error in sync PDF text extraction: {str(e)}")
            raise
    
    def _ocr_page(self, page, page_num: int) -> str:
        """Extract text from a single PDF page using OCR"""
        try:
            # Convert page to image with higher resolution for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Enhance image for better OCR results
            img = self._enhance_image_for_ocr(img)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                img,
                lang=self.tesseract_config['lang'],
                config=self.tesseract_config['config']
            )
            
            # Clean up
            pix = None
            
            return ocr_text.strip()
            
        except Exception as e:
            logger.error(f"Error in OCR for page {page_num}: {str(e)}")
            return f"[OCR Error on page {page_num}: {str(e)}]"
    
    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results"""
        try:
            # Convert to grayscale if not already
            if img.mode != 'L':
                img = img.convert('L')
            
            # You can add more enhancement techniques here:
            # - Noise reduction
            # - Contrast adjustment
            # - Deskewing
            # - Binarization
            
            return img
            
        except Exception as e:
            logger.warning(f"Error enhancing image for OCR: {str(e)}")
            return img
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image file using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text string
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._extract_text_from_image_sync,
                image_path
            )
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {str(e)}")
            return f"Error extracting text from image: {str(e)}"
    
    def _extract_text_from_image_sync(self, image_path: str) -> str:
        """Synchronous image text extraction"""
        try:
            # Open and enhance image
            img = Image.open(image_path)
            img = self._enhance_image_for_ocr(img)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                img,
                lang=self.tesseract_config['lang'],
                config=self.tesseract_config['config']
            )
            
            return text.strip() if text.strip() else "No text detected in image."
            
        except Exception as e:
            logger.error(f"Error in sync image text extraction: {str(e)}")
            raise
    
    def get_ocr_confidence(self, image_path: str) -> Dict:
        """Get OCR confidence data for image"""
        try:
            img = Image.open(image_path)
            img = self._enhance_image_for_ocr(img)
            
            # Get detailed OCR data with confidence scores
            data = pytesseract.image_to_data(
                img,
                lang=self.tesseract_config['lang'],
                config=self.tesseract_config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'average_confidence': avg_confidence,
                'word_count': len([w for w in data['text'] if w.strip()]),
                'low_confidence_words': len([c for c in confidences if c < 60]),
                'high_confidence_words': len([c for c in confidences if c >= 80])
            }
            
        except Exception as e:
            logger.error(f"Error getting OCR confidence: {str(e)}")
            return {'error': str(e)}
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)