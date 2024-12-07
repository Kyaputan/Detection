from ultralytics import YOLO
import os


model_path = "Collet\Model\last.pt"
image_paths = ["Lab1 Detection\ex01\ld1lDGtifuOjZrQBqZ4L.webp"]
print("Test1")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist")


model = YOLO(model_path)
print("Test2")

for image_path in image_paths:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist")
print("Test3")


model.info()
for image_path in image_paths:
    results = model.predict(source=image_path)
    for result in results:
        result.show()  # แสดงผลลัพธ์
        result.save("Output")  # บันทึกผลลัพธ์ไปยังโฟลเดอร์ "Output"



