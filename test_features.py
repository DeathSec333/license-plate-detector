from plate_detector import LicensePlateDetector

detector = LicensePlateDetector()

print("🧪 Testing new features...")

# Test benchmark
print("\n📊 Benchmark Test:")
benchmark = detector.benchmark_methods('test_images/test_car.jpg')
for method, stats in benchmark.items():
    print(f"🔧 {method}: {stats['time']:.3f}s, {stats['detections']} detections")

# Test live detection
print("\n🎥 Live Detection Test:")
detector.live_detection()

print("\n✅ Feature tests completed!")
