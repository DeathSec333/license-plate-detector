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
