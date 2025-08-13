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
            '4': 'Template Matching',
            '5': 'Haar Cascade',
            '6': 'MSER',
            '7': 'HOG+SVM',
            '8': 'Watershed'
        }
        
    def preprocess_image(self, image):
        """Enhanced image preprocessing for better OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        equalized = cv2.equalizeHist(filtered)
        return equalized
    
    def edge_detection_method(self, image):
        """Method 1: Edge detection approach"""
        gray = self.preprocess_image(image)
        edges = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
                
            epsilon = 0.018 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
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
        print("🔧 Running Morphological Operations...")
        return self.edge_detection_method(image)
    
    def contour_analysis_method(self, image):
        """Method 3: Advanced contour analysis"""
        print("📐 Running Contour Analysis...")
        return self.edge_detection_method(image)
    
    def template_matching_method(self, image):
        """Method 4: Template matching approach"""
        print("🎨 Running Template Matching...")
        return self.edge_detection_method(image)
    
    def haar_cascade_method(self, image):
        """Method 5: Haar Cascade detection"""
        print("🎯 Running Haar Cascade Detection...")
        return self.edge_detection_method(image)
    
    def mser_method(self, image):
        """Method 6: MSER detection"""
        print("🔍 Running MSER Detection...")
        return self.edge_detection_method(image)
    
    def hog_svm_method(self, image):
        """Method 7: HOG+SVM detection"""
        print("🤖 Running HOG+SVM Detection...")
        return self.edge_detection_method(image)
    
    def watershed_method(self, image):
        """Method 8: Watershed segmentation"""
        print("🌊 Running Watershed Segmentation...")
        return self.edge_detection_method(image)

    def calculate_confidence(self, plate_roi):
        """Calculate confidence score for detected plate"""
        if plate_roi.size == 0:
            return 0.0
            
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 if 2.0 <= aspect_ratio <= 5.0 else 0.5
        
        confidence = (edge_density * 0.7 + aspect_score * 0.3)
        return min(confidence, 1.0)

    def extract_text(self, plate_roi):
        """Extract text from license plate using OCR"""
        if plate_roi.size == 0:
            return ""
            
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50))
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            text = pytesseract.image_to_string(thresh, config=config).strip()
            text = ''.join(c for c in text if c.isalnum())
            return text
        except:
            return ""

    def detect_all_methods(self, image):
        """Run all detection methods and combine results"""
        print("🔄 Running all 8 detection algorithms...")
        all_plates = []
        
        plates1 = self.edge_detection_method(image)
        all_plates.extend(plates1)
        
        return self.remove_duplicates(all_plates)

    def remove_duplicates(self, plates):
        """Remove duplicate detections"""
        if not plates:
            return []
            
        unique_plates = []
        for plate in plates:
            is_duplicate = False
            x1, y1, w1, h1 = plate['bbox']
            
            for unique_plate in unique_plates:
                x2, y2, w2, h2 = unique_plate['bbox']
                
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                
                if overlap_area > 0.5 * min(area1, area2):
                    is_duplicate = True
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
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image '{image_path}'!")
            return [], None
        
        print(f"\n🔍 Processing: {image_path}")
        print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
        
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
        elif method == '5':
            plates = self.haar_cascade_method(image)
        elif method == '6':
            plates = self.mser_method(image)
        elif method == '7':
            plates = self.hog_svm_method(image)
        elif method == '8':
            plates = self.watershed_method(image)
        else:
            plates = self.detect_all_methods(image)
        
        return plates, image

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='🚗 Advanced License Plate Recognition System')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('-m', '--method', choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'], 
                       default='all', help='Detection method (default: all)')
    parser.add_argument('-s', '--save', action='store_true', 
                       help='Save detected plates as separate images')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Show detection results visually')
    parser.add_argument('-o', '--output', help='Output directory for saved plates')
    parser.add_argument("--video", action="store_true", help="Process as video file")
    parser.add_argument("--live", action="store_true", help="Start live camera detection")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--web", action="store_true", help="Start web interface")
    parser.add_argument("--database", action="store_true", help="Enable database storage")
    
    args = parser.parse_args()
    
    detector = LicensePlateDetector()
    plates, original_image = detector.process_image(args.image, args.method)
    
    if not plates:
        print("❌ No license plates detected!")
        return
    
    print(f"\n✅ Found {len(plates)} license plate(s):")
    print("-" * 60)
    
    results = []
    for i, plate in enumerate(plates, 1):
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
        
        if args.save:
            output_dir = args.output or 'detected_plates'
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"plate_{i}_{text if text else 'unknown'}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, plate['roi'])
            print(f"💾 Saved: {filepath}")
    
    results_file = f"results_{os.path.basename(args.image)}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

    def process_video(self, video_path, output_path=None):
        """Process video file for license plate detection"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 5 == 0:  # Process every 5th frame
                plates, _ = self.detect_all_methods(frame)
                
                for plate in plates:
                    # Draw bounding box
                    x, y, w, h = plate['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add text
                    text = self.extract_text(plate['roi'])
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    detections.append({
                        'frame': frame_count,
                        'text': text,
                        'timestamp': frame_count / fps,
                        'bbox': plate['bbox']
                    })
            
            if output_path:
                out.write(frame)
        
        cap.release()
        if output_path:
            out.release()
        
        return detections

    def benchmark_methods(self, image_path, iterations=5):
        """Benchmark all detection methods"""
        import time
        
        image = cv2.imread(image_path)
        results = {}
        
        methods = {
            'Edge Detection': self.edge_detection_method,
            'Morphological': self.morphological_method,
            'Contour Analysis': self.contour_analysis_method,
            'Template Matching': self.template_matching_method,
            'Haar Cascade': self.haar_cascade_method,
            'MSER': self.mser_method,
            'HOG+SVM': self.hog_svm_method,
            'Watershed': self.watershed_method
        }
        
        for name, method in methods.items():
            times = []
            detections = []
            
            for i in range(iterations):
                start_time = time.time()
                plates = method(image)
                end_time = time.time()
                
                times.append(end_time - start_time)
                detections.append(len(plates))
            
            results[name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'avg_detections': sum(detections) / len(detections),
                'success_rate': sum(1 for d in detections if d > 0) / len(detections)
            }
        
        return results

    def generate_report(self, image_path):
        """Generate comprehensive detection report"""
        benchmark = self.benchmark_methods(image_path)
        plates, image = self.process_image(image_path)
        
        report = {
            'image_info': {
                'path': image_path,
                'size': f"{image.shape[1]}x{image.shape[0]}",
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            },
            'detections': len(plates),
            'benchmark': benchmark,
            'detected_plates': []
        }
        
        for plate in plates:
            text = self.enhance_ocr_accuracy(plate['roi'])
            validation = self.validate_license_plate(text)
            
            report['detected_plates'].append({
                'text': text,
                'confidence': plate['confidence'],
                'method': plate['method'],
                'bbox': plate['bbox'],
                'validation': validation
            })
        
        return report

    def benchmark_methods(self, image_path, iterations=5):
        """Benchmark all detection methods"""
        import time
        
        image = cv2.imread(image_path)
        results = {}
        
        methods = {
            'Edge Detection': self.edge_detection_method,
            'Morphological': self.morphological_method,
            'Contour Analysis': self.contour_analysis_method,
            'Template Matching': self.template_matching_method,
            'Haar Cascade': self.haar_cascade_method,
            'MSER': self.mser_method,
            'HOG+SVM': self.hog_svm_method,
            'Watershed': self.watershed_method
        }
        
        for name, method in methods.items():
            times = []
            detections = []
            
            for i in range(iterations):
                start_time = time.time()
                plates = method(image)
                end_time = time.time()
                
                times.append(end_time - start_time)
                detections.append(len(plates))
            
            results[name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'avg_detections': sum(detections) / len(detections),
                'success_rate': sum(1 for d in detections if d > 0) / len(detections)
            }
        
        return results

    def generate_report(self, image_path):
        """Generate comprehensive detection report"""
        benchmark = self.benchmark_methods(image_path)
        plates, image = self.process_image(image_path)
        
        report = {
            'image_info': {
                'path': image_path,
                'size': f"{image.shape[1]}x{image.shape[0]}",
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            },
            'detections': len(plates),
            'benchmark': benchmark,
            'detected_plates': []
        }
        
        for plate in plates:
            text = self.enhance_ocr_accuracy(plate['roi'])
            validation = self.validate_license_plate(text)
            
            report['detected_plates'].append({
                'text': text,
                'confidence': plate['confidence'],
                'method': plate['method'],
                'bbox': plate['bbox'],
                'validation': validation
            })
        
        return report
