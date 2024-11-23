from ultralytics import YOLO
import cv2
import os

# ตรวจสอบว่าพาธที่ใช้ถูกต้องหรือไม่
model_path = "Lab1 Detection\ex01\last.pt"
image_path = "Lab1 Detection\ex01\ld1lDGtifuOjZrQBqZ4L.webp"

# ตรวจสอบการมีอยู่ของไฟล์โมเดล
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist")

# โหลดโมเดลที่เทรนแล้ว
model = YOLO(model_path)

# ตรวจสอบการมีอยู่ของไฟล์รูปภาพ
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image file {image_path} does not exist")

# ฟังก์ชันสำหรับวาดกรอบและข้อความบนภาพ
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.data.cpu().numpy()  # เอาข้อมูล bounding boxes ออกมา
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f'{result.names[class_id]} : {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# ใช้โมเดลในการตรวจจับวัตถุในภาพ
image = cv2.imread(image_path)
results = model.predict(source=image)

# วาดกรอบและตรวจสอบว่าพบรถแท็กซี่หรือไม่
draw_boxes(image, results)

# แสดงผลลัพธ์
cv2.imshow("Detected Image", image)
cv2.waitKey(0)  # กดปุ่มใด ๆ เพื่อปิดหน้าต่างแสดงผล
cv2.destroyAllWindows()

# บันทึกผลลัพธ์ไปยังโฟลเดอร์ "Output"
output_path = os.path.join("Output", os.path.basename(image_path))
os.makedirs("Output", exist_ok=True)
cv2.imwrite(output_path, image)