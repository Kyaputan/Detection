from ultralytics import YOLO
import os

model_paths = [
    "Collet/Model/best (1).pt",
    "Collet/Model/best.pt",
    "Collet/Model/last.pt"
]

image_paths = ["Lab1/ex01/ld1lDGtifuOjZrQBqZ4L.webp","Collet/Picture/fd2b5822-7f1c-48c5-9ed8-651e10470a6c.jpeg"]


for model_path in model_paths:
    if not os.path.exists(model_path):
        print(f"The model file {model_path} does not exist")


for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"The image file {image_path} does not exist")

models = [(YOLO(model_path), model_path) for model_path in model_paths]

for idx, (model, model_path) in enumerate(models, start=1):
    print("\n")
    print(f"Using model #{idx}: {model_path}")
    model.info()
    for image_path in image_paths:
        results = model.predict(source=image_path)
        for result in results:
            result.show() 
            output_dir = os.path.join("Output", os.path.splitext(os.path.basename(model_path))[0])
            os.makedirs(output_dir, exist_ok=True)
            result.save(output_dir) 

