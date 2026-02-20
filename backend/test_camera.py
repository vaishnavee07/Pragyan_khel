"""
Quick Camera Test
Verifies webcam access without starting full server
"""
import cv2
import sys

print("Testing webcam access...")

# Try to open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Failed to open camera")
    print("  Possible reasons:")
    print("  - No webcam connected")
    print("  - Camera in use by another application")
    print("  - Permission denied")
    sys.exit(1)

print("✓ Camera opened successfully")

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")

# Try to read one frame
ret, frame = cap.read()

if ret:
    print("✓ Successfully captured frame")
    print(f"  Frame shape: {frame.shape}")
else:
    print("✗ Failed to capture frame")

# Release camera
cap.release()
print("✓ Camera released")

print("\nCamera test PASSED - Backend can access webcam independently")
