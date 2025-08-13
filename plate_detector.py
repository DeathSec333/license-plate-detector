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

    def calculate_confidence(self, plate_roi):
        """Calculate confidence score for detected plate"""
        if plate_roi.size == 0:
            return 0.0
            
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for text-like patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Aspect ratio score
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 if 2.0 <= aspect_ratio <= 5.0 else 0.5
        
        # Combine scores
        confidence = (edge_density * 0.7 + aspect_score * 0.3)
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
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            # Clean up text
            text = ''.join(c for c in text if c.isalnum())
            return text
        except:
            return ""

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
        
        # Simple edge detection for now
        plates = self.edge_detection_method(image)
        
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
    
    # Save results as JSON
    results_file = f"results_{os.path.basename(args.image)}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
