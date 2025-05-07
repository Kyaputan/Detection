from ultralytics import YOLO
import cv2
import os

folder_path = os.path.dirname(os.path.realpath(__file__))
modelYolo = os.path.join(folder_path, "yolo11n-cls.pt")
model = YOLO(modelYolo)
print("=="*70)
print("info : " ,model.info())
print("=="*70)
print("names:", model.names)
print("=="*70)
