#!/usr/bin/env python3
"""
Advanced License Plate Recognition System
Multi-algorithm detection with OCR processing
"""

import cv2
import numpy as np
import pytesseract
import os
import json
from datetime import datetime
import argparse

class LicensePlateDetector:
    def __init__(self):
        self.detection_methods = {
            '1': 'Edge Detection',
            '2': 'Morphological Operations', 
            '3': 'Contour Analysis',
            '4': 'Template Matching'
        }
        
    def preprocess_image(self, image):
        """Enhanced image preprocessing for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(filtered)
        
        return equalized
    
    def edge_detection_method(self, image):
        """Method 1: Edge detection approach"""
        gray = self.preprocess_image(image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            # Calculate contour area and perimeter
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            # Approximate contour to polygon
            epsilon = 0.018 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if contour has 4 corners (rectangular)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate aspect ratio check
                if 2.0 <= aspect_ratio <= 5.0:
                    plate_roi = image[y:y+h, x:x+w]
                    plates.append({
                        'roi': plate_roi,
                        'bbox': (x, y, w, h),
                        'method': 'Edge Detection',
                        'confidence': self.calculate_confidence(plate_roi)
                    })
        
        return plates
    
    def morphological_method(self, image):
        """Method 2: Morphological operations approach"""
        gray = self.preprocess_image(image)
        
        # Apply morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Apply threshold
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply closing operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = w * h
            
            # Filter based on size and aspect ratio
            if 2000 < area < 50000 and 2.0 <= aspect_ratio <= 5.0:
                plate_roi = image[y:y+h, x:x+w]
                plates.append({
                    'roi': plate_roi,
                    'bbox': (x, y, w, h),
                    'method': 'Morphological',
                    'confidence': self.calculate_confidence(plate_roi)
                })
        
        return plates
    
    def contour_analysis_method(self, image):
        """Method 3: Advanced contour analysis"""
        gray = self.preprocess_image(image)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check for license plate characteristics
            if 2.0 <= aspect_ratio <= 5.0 and 1000 < area < 50000:
                # Additional validation
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity > 0.6:  # License plates are usually solid rectangles
                    plate_roi = image[y:y+h, x:x+w]
                    plates.append({
                        'roi': plate_roi,
                        'bbox': (x, y, w, h),
                        'method': 'Contour Analysis',
                        'confidence': self.calculate_confidence(plate_roi)
                    })
        
        return plates
    
    def template_matching_method(self, image):
        """Method 4: Template matching approach"""
        gray = self.preprocess_image(image)
        
        # Create synthetic license plate template
        template = np.ones((60, 240), dtype=np.uint8) * 255
        cv2.rectangle(template, (10, 10), (230, 50), 0, 2)
        
        # Template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.3)
        
        plates = []
        for pt in zip(*locations[::-1]):
            x, y = pt
            w, h = template.shape[::-1]
            
            # Extract potential plate region
            plate_roi = image[y:y+h, x:x+w]
            plates.append({
                'roi': plate_roi,
                'bbox': (x, y, w, h),
                'method': 'Template Matching',
                'confidence': result[y, x]
            })
        
        return plates
    
    def calculate_confidence(self, plate_roi):

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        """Calculate confidence score for detected plate"""

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        if plate_roi.size == 0:

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
            return 0.0

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
            

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        # Check for text-like patterns

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        edges = cv2.Canny(gray, 50, 150)

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        edge_density = np.sum(edges > 0) / edges.size

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        # Aspect ratio score

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        h, w = gray.shape

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        aspect_ratio = w / h if h > 0 else 0

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        aspect_score = 1.0 if 2.0 <= aspect_ratio <= 5.0 else 0.5

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        # Combine scores

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        confidence = (edge_density * 0.7 + aspect_score * 0.3)

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
        return min(confidence, 1.0)

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        # Preprocess for OCR
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR configuration
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = "".join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        all_plates = []
        
        # Method 1: Edge Detection
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        # Method 2: Morphological
        plates2 = self.morphological_method(image)
        all_plates.extend(plates2)
        
        # Method 3: Contour Analysis
        plates3 = self.contour_analysis_method(image)
        all_plates.extend(plates3)
        
        # Method 4: Template Matching
        plates4 = self.template_matching_method(image)
        all_plates.extend(plates4)
        
        # Remove duplicates and sort by confidence
        unique_plates = self.remove_duplicates(all_plates)
        unique_plates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_plates
    
    def remove_duplicates(self, plates):
        """Remove duplicate detections"""
        if not plates:
            return [], None
            
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            x1, y1, w1, h1 = plate['bbox']
            
            for unique_plate in unique_plates:
                x2, y2, w2, h2 = unique_plate['bbox']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If significant overlap, consider as duplicate
                if overlap_area > 0.5 * min(area1, area2):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if plate['confidence'] > unique_plate['confidence']:
                        unique_plates.remove(unique_plate)
                        unique_plates.append(plate)
                    break
            
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates
    
    def process_image(self, image_path, method='all'):
        """Process single image for license plate detection"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return [], None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image '{image_path}'!")
            return [], None
        
        print(f"\n🔍 Processing: {image_path}")
        print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect plates based on method
        if method == 'all':
            plates = self.detect_all_methods(image)
        elif method == '1':
            plates = self.edge_detection_method(image)
        elif method == '2':
            plates = self.morphological_method(image)
        elif method == '3':
            plates = self.contour_analysis_method(image)
        elif method == '4':
            plates = self.template_matching_method(image)
        else:
            plates = self.detect_all_methods(image)
        
        return plates, image

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='🚗 Advanced License Plate Recognition System')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-m', '--method', choices=['1', '2', '3', '4', 'all'], 
                       default='all', help='Detection method (default: all)')
    parser.add_argument('-s', '--save', action='store_true', 
                       help='Save detected plates as separate images')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Show detection results visually')
    parser.add_argument('-o', '--output', help='Output directory for saved plates')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Process image
    plates, original_image = detector.process_image(args.image, args.method)
    
    if not plates:
        print("❌ No license plates detected!")
        return
    
    print(f"\n✅ Found {len(plates)} license plate(s):")
    print("-" * 60)
    
    results = []
    for i, plate in enumerate(plates, 1):
        # Extract text
        text = detector.extract_text(plate['roi'])
        
        result = {
            'plate_number': i,
            'text': text if text else 'No text detected',
            'method': plate['method'],
            'confidence': plate['confidence'],
            'bbox': plate['bbox'],
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"🔢 Plate #{i}:")
        print(f"   📝 Text: {result['text']}")
        print(f"   🔧 Method: {result['method']}")
        print(f"   📊 Confidence: {result['confidence']:.3f}")
        print(f"   📍 Location: {result['bbox']}")
        print()
        
        # Save individual plates if requested
        if args.save:
            output_dir = args.output or 'detected_plates'
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"plate_{i}_{text if text else 'unknown'}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, plate['roi'])
            print(f"💾 Saved: {filepath}")
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(original_image, plates, results)
    
    # Save results as JSON
    results_file = f"results_{os.path.basename(args.image)}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {results_file}")

def visualize_results(image, plates, results):
    """Visualize detection results"""
    display_image = image.copy()
    
    for i, (plate, result) in enumerate(zip(plates, results)):
        x, y, w, h = plate['bbox']
        
        # Draw bounding box
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        label = f"{i+1}: {result['text']} ({result['confidence']:.2f})"
        cv2.putText(display_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display result
    cv2.imshow('License Plate Detection Results', display_image)
    print("🖼️  Press any key to close the visualization window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
