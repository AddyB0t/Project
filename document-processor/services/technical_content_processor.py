#!/usr/bin/env python3
"""
Technical Content Processor

Converts technical drawings and numeric data into semantic, LLM-friendly content.
Specializes in architectural drawings, floor plans, and technical specifications.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RoomInfo:
    """Information about a detected room"""
    name: str
    room_type: str
    mentions: int
    context: List[str]

@dataclass
class TechnicalContent:
    """Processed technical content"""
    original_text: str
    semantic_description: str
    room_info: List[RoomInfo]
    spatial_elements: List[str]
    measurements: List[str]
    technical_features: List[str]

class TechnicalContentProcessor:
    """
    Processes technical content to extract semantic meaning
    """
    
    def __init__(self):
        self.room_patterns = self._compile_room_patterns()
        self.measurement_patterns = self._compile_measurement_patterns()
        self.spatial_patterns = self._compile_spatial_patterns()
    
    def _compile_room_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for room detection"""
        patterns = {
            'office': re.compile(r'\bOFFICE\s*(\d+)?\b', re.IGNORECASE),
            'reception': re.compile(r'\bRECEPTION\b', re.IGNORECASE),
            'room': re.compile(r'\bROOM\s*(\d+)?\b', re.IGNORECASE),
            'bedroom': re.compile(r'\bBEDROOM\s*(\d+)?\b', re.IGNORECASE),
            'bathroom': re.compile(r'\b(BATHROOM|WASHROOM|WC|TOILET)\s*(\d+)?\b', re.IGNORECASE),
            'kitchen': re.compile(r'\bKITCHEN\b', re.IGNORECASE),
            'living': re.compile(r'\bLIVING\s*ROOM\b', re.IGNORECASE),
            'dining': re.compile(r'\bDINING\s*ROOM\b', re.IGNORECASE),
            'conference': re.compile(r'\bCONFERENCE\s*ROOM\b', re.IGNORECASE),
            'meeting': re.compile(r'\bMEETING\s*ROOM\b', re.IGNORECASE),
            'storage': re.compile(r'\b(STORAGE|STORE)\s*ROOM?\b', re.IGNORECASE),
            'lobby': re.compile(r'\bLOBBY\b', re.IGNORECASE),
            'hall': re.compile(r'\bHALL\b', re.IGNORECASE),
            'corridor': re.compile(r'\b(CORRIDOR|PASSAGE)\b', re.IGNORECASE)
        }
        return patterns
    
    def _compile_measurement_patterns(self) -> List[re.Pattern]:
        """Compile patterns for measurement detection"""
        return [
            re.compile(r'\b\d{3,5}\s*(?:mm|cm|m)\b', re.IGNORECASE),  # Measurements
            re.compile(r'\b\d{1,4}\.\d{1,2}\s*(?:mm|cm|m)\b', re.IGNORECASE),  # Decimal measurements
            re.compile(r'\b\d{3,5}\s*x\s*\d{3,5}\b'),  # Dimensions (e.g., 2100 x 900)
        ]
    
    def _compile_spatial_patterns(self) -> List[re.Pattern]:
        """Compile patterns for spatial elements"""
        return [
            re.compile(r'\b(DOOR|WINDOW|WALL|PARTITION)\b', re.IGNORECASE),
            re.compile(r'\b(ENTRANCE|EXIT|STAIR|ELEVATOR)\b', re.IGNORECASE),
            re.compile(r'\bGREEN\s*SPACE\b', re.IGNORECASE),
            re.compile(r'\b(THICK|SOLID|BLOCK)\s*WALL\b', re.IGNORECASE),
        ]
    
    def process_technical_text(self, text: str) -> TechnicalContent:
        """
        Main processing function - converts technical text to semantic content
        
        Args:
            text: Raw technical text from OCR
            
        Returns:
            TechnicalContent with semantic interpretation
        """
        logger.info(f"Processing technical text: {len(text)} characters")
        
        # Extract room information
        room_info = self._extract_room_info(text)
        
        # Extract spatial elements
        spatial_elements = self._extract_spatial_elements(text)
        
        # Extract measurements
        measurements = self._extract_measurements(text)
        
        # Extract technical features
        technical_features = self._extract_technical_features(text)
        
        # Generate semantic description
        semantic_description = self._generate_semantic_description(
            room_info, spatial_elements, measurements, technical_features
        )
        
        return TechnicalContent(
            original_text=text,
            semantic_description=semantic_description,
            room_info=room_info,
            spatial_elements=spatial_elements,
            measurements=measurements,
            technical_features=technical_features
        )
    
    def _extract_room_info(self, text: str) -> List[RoomInfo]:
        """Extract room information from text"""
        rooms = []
        room_contexts = {}
        
        # Split text into lines for context
        lines = text.split('\n')
        
        for room_type, pattern in self.room_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Get context around matches
                context_lines = []
                for line in lines:
                    if pattern.search(line):
                        context_lines.append(line.strip())
                
                room_info = RoomInfo(
                    name=room_type.title(),
                    room_type=room_type,
                    mentions=len(matches),
                    context=context_lines
                )
                rooms.append(room_info)
                
                logger.info(f"Found {len(matches)} {room_type} mentions")
        
        return rooms
    
    def _extract_spatial_elements(self, text: str) -> List[str]:
        """Extract spatial elements like doors, windows, walls"""
        elements = []
        
        for pattern in self.spatial_patterns:
            matches = pattern.findall(text)
            elements.extend(matches)
        
        # Remove duplicates while preserving order
        unique_elements = list(dict.fromkeys(elements))
        
        if unique_elements:
            logger.info(f"Found spatial elements: {unique_elements}")
        
        return unique_elements
    
    def _extract_measurements(self, text: str) -> List[str]:
        """Extract measurements and dimensions"""
        measurements = []
        
        for pattern in self.measurement_patterns:
            matches = pattern.findall(text)
            measurements.extend(matches)
        
        # Remove duplicates and limit to most relevant
        unique_measurements = list(dict.fromkeys(measurements))[:10]  # Limit to 10 most relevant
        
        if unique_measurements:
            logger.info(f"Found measurements: {unique_measurements[:5]}...")  # Log first 5
        
        return unique_measurements
    
    def _extract_technical_features(self, text: str) -> List[str]:
        """Extract technical features and specifications"""
        features = []
        
        # Look for wall specifications
        wall_specs = re.findall(r'\b\d+\s*mm\s*\w+\s*wall\b', text, re.IGNORECASE)
        features.extend(wall_specs)
        
        # Look for EPS wall specifications
        eps_specs = re.findall(r'\b\d+mm\s*EPS\s*wall\b', text, re.IGNORECASE)
        features.extend(eps_specs)
        
        # Look for block wall specifications
        block_specs = re.findall(r'\bThick\s*solid\s*block\s*wall\b', text, re.IGNORECASE)
        features.extend(block_specs)
        
        if features:
            logger.info(f"Found technical features: {features}")
        
        return features
    
    def _generate_semantic_description(self, 
                                     room_info: List[RoomInfo], 
                                     spatial_elements: List[str],
                                     measurements: List[str], 
                                     technical_features: List[str]) -> str:
        """Generate a human-readable semantic description"""
        
        description_parts = []
        
        # Describe the building/layout type
        if room_info:
            room_types = [room.room_type for room in room_info]
            if 'office' in room_types:
                description_parts.append("This appears to be an office building or commercial space floor plan.")
            elif any(room_type in ['bedroom', 'bathroom', 'kitchen', 'living'] for room_type in room_types):
                description_parts.append("This appears to be a residential floor plan.")
            else:
                description_parts.append("This is an architectural floor plan showing various rooms and spaces.")
        
        # Describe rooms found
        if room_info:
            total_rooms = sum(room.mentions for room in room_info)
            room_names = []
            
            for room in room_info:
                if room.mentions > 1:
                    room_names.append(f"{room.mentions} {room.name.lower()}s")
                else:
                    room_names.append(f"1 {room.name.lower()}")
            
            if room_names:
                rooms_text = ", ".join(room_names)
                description_parts.append(f"The layout contains {rooms_text}.")
                
                # Add specific room details
                for room in room_info:
                    if room.room_type == 'office' and room.mentions > 1:
                        description_parts.append(f"There are multiple office spaces numbered from 1 to {room.mentions}.")
                    elif room.room_type == 'reception':
                        description_parts.append("There is a reception area at the entrance.")
        
        # Describe spatial features
        if spatial_elements:
            unique_elements = list(set([elem.lower() for elem in spatial_elements]))
            if len(unique_elements) > 0:
                elements_text = ", ".join(unique_elements)
                description_parts.append(f"The plan includes architectural elements such as {elements_text}.")
        
        # Describe technical specifications
        if technical_features:
            description_parts.append("Technical specifications include various wall types and construction details.")
        
        # Add measurement context
        if measurements:
            description_parts.append(f"The drawing includes detailed measurements and dimensions ({len(measurements)} measurement annotations).")
        
        # Default description if nothing found
        if not description_parts:
            description_parts.append("This is a technical architectural drawing with detailed specifications and measurements.")
        
        final_description = " ".join(description_parts)
        logger.info(f"Generated semantic description: {final_description[:100]}...")
        
        return final_description

    def enhance_chunk_content(self, chunk_text: str, chunk_type: str = 'text') -> str:
        """
        Enhance a text chunk with semantic content for better LLM understanding
        
        Args:
            chunk_text: Original chunk text
            chunk_type: Type of chunk ('text' or 'image')
            
        Returns:
            Enhanced text with semantic meaning
        """
        if chunk_type == 'image':
            return chunk_text  # Images are processed differently
        
        # Process the technical content
        processed = self.process_technical_text(chunk_text)
        
        # Create enhanced content that combines original with semantic meaning
        enhanced_parts = []
        
        # Add semantic description first
        if processed.semantic_description:
            enhanced_parts.append(f"SEMANTIC CONTENT: {processed.semantic_description}")
        
        # Add room information
        if processed.room_info:
            room_descriptions = []
            for room in processed.room_info:
                room_desc = f"{room.name}"
                if room.mentions > 1:
                    room_desc += f" (appears {room.mentions} times)"
                room_descriptions.append(room_desc)
            
            enhanced_parts.append(f"ROOMS IDENTIFIED: {', '.join(room_descriptions)}")
        
        # Add spatial context
        if processed.spatial_elements:
            enhanced_parts.append(f"SPATIAL ELEMENTS: {', '.join(processed.spatial_elements)}")
        
        # Add original technical content for reference
        enhanced_parts.append(f"TECHNICAL DATA: {chunk_text}")
        
        enhanced_content = "\n\n".join(enhanced_parts)
        
        logger.info(f"Enhanced chunk from {len(chunk_text)} to {len(enhanced_content)} characters")
        
        return enhanced_content

# Global instance for easy access
_technical_processor = None

def get_technical_processor() -> TechnicalContentProcessor:
    """Get singleton instance of technical processor"""
    global _technical_processor
    if _technical_processor is None:
        _technical_processor = TechnicalContentProcessor()
        logger.info("✅ Technical content processor initialized")
    return _technical_processor