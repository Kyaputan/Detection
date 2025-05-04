from ultralytics import YOLO
model = YOLO('yolov8x.pt')
print("Classes:", model.names)