#!/bin/bash
echo "🚗 Installing License Plate Recognition System..."

# Update system packages
echo "📦 Updating system packages..."
pkg update -y
pkg upgrade -y

# Install required system packages
echo "🔧 Installing system dependencies..."
pkg install -y python python-pip tesseract opencv-python

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p detected_plates
mkdir -p test_images
mkdir -p results

echo "✅ Installation complete!"
echo "🚀 Usage: python plate_detector.py <image_path>"
echo "📖 Example: python plate_detector.py test_images/car.jpg -v -s"
