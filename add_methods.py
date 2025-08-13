# Read the current file
with open('plate_detector.py', 'r') as f:
    content = f.read()

# Find where to insert (before "def main():")
main_pos = content.find('def main():')

# The methods to add (with proper indentation)
new_methods = '''
    def live_detection(self, camera_id=0):
        """Real-time license plate detection simulation"""
        print("📱 Simulating live detection...")
        
        for i in range(3):
            print(f"📸 Frame {i+1}: Processing...")
            plates, _ = self.process_image('test_images/test_car.jpg')
            if plates:
                text = self.extract_text(plates[0]['roi'])
                print(f"   🎯 Detected: {text}")
            import time
            time.sleep(1)

    def benchmark_methods(self, image_path, iterations=3):
        """Benchmark detection methods"""
        import time
        
        if not os.path.exists(image_path):
            return {}
            
        image = cv2.imread(image_path)
        results = {}
        
        methods = {
            'Edge Detection': self.edge_detection_method,
            'Haar Cascade': self.haar_cascade_method,
            'MSER': self.mser_method,
        }
        
        for name, method in methods.items():
            start_time = time.time()
            plates = method(image)
            end_time = time.time()
            
            results[name] = {
                'time': end_time - start_time,
                'detections': len(plates)
            }
        
        return results

    def save_to_database(self, result):
        """Save to database simulation"""
        print(f"💾 Database: Saved '{result['text']}'")

'''

# Insert the methods before main function
new_content = content[:main_pos] + new_methods + content[main_pos:]

# Write back to file
with open('plate_detector.py', 'w') as f:
    f.write(new_content)

print("✅ Methods added successfully!")
