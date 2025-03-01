import cv2
import glob
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# โหลดโมเดล YOLO
model = YOLO("D:\Code\CodeCit\Lab1-Detection\ex14_blurring\Blur.pt")
names = model.names

# กำหนดพาธของโฟลเดอร์ที่มีรูปภาพ
image_folder = "CodeCit\Lab1-Detection\ex14_blurring\input"
output_folder = "CodeCit\Lab1-Detection\ex14_blurring\Output"
os.makedirs(output_folder, exist_ok=True)

# ดึงชื่อไฟล์รูปภาพทั้งหมดที่เป็น .jpg
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

# Blur ratio
blur_ratio = 50

for image_path in image_files:
    im0 = cv2.imread(image_path)
    if im0 is None:
        print(f"Error reading image: {image_path}")
        continue

    results = model.predict(im0 , conf = 0.01)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj

    # บันทึกภาพผลลัพธ์
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, im0)
    print(f"Processed and saved: {output_path}")

print("Processing complete.")