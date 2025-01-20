from ultralytics import YOLO
import os

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "model.pt")

print("folder_path: ", folder_path)
print("model_path: ", model_path)



