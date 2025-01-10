from ultralytics import YOLO
import os

# folder_path = os.path.dirname(os.path.realpath(__file__))
# model_path = os.path.join(folder_path, "model.pt")
# model = YOLO(model_path)
model = YOLO('yolo11n.pt')
model.export(format='onnx',simplify=True,int8=True,imgsz=640) #ONNX for CPU , TensorRT for GPU 

