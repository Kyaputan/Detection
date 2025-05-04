import cv2
import numpy as np

roi_width = 150
roi_height = 150

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตั้งค่าความละเอียดของกล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX  # กำหนดฟอนต์


def calculate_average_rgb(roi):
    """
    ฟังก์ชันนี้รับภาพ ROI และส่งคืนค่าเฉลี่ยสี RGB

    Args:
    roi: รูปภาพ ROI

    Returns:
    tuple: ค่าเฉลี่ยสี RGB (B, G, R)
    """
    # แปลงภาพเป็นระบบสี BGR
    bgr_mean = cv2.mean(roi)

    # แยกค่าสี BGR ออกมา
    blue, green, red = bgr_mean[0], bgr_mean[1], bgr_mean[2]

    # ส่งคืนค่าเฉลี่ยสี RGB
    return blue, green, red


while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()

    # ตรวจสอบว่ามีเฟรมหรือไม่
    if not ret:
        break

    # แปลงภาพเป็นระบบสี BGR
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # กำหนดตำแหน่ง ROI
    x = frame.shape[1] // 2 - roi_width // 2
    y = frame.shape[0] // 2 - roi_height // 2

    # สร้าง ROI
    roi = bgr_frame[y : y + roi_height, x : x + roi_width]

    # คำนวณค่าเฉลี่ยสี RGB
    blue, green, red = calculate_average_rgb(roi)

    # แสดงค่าสี RGB บนหน้าจอ
    cv2.putText(frame, f"R: {green:.4f} %", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"G: {blue:.4f} %", (10, 90), font, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"B: {red:.4f} %", (10, 60), font, 0.7, (255, 0, 0), 2)

    if blue >= 150 and blue > green and blue > red:
        color_name = "green"
    elif green >= 150 and green > blue and green > red:
        color_name = "red"
    elif red >= 150 and red > blue and red > green:
        color_name = "blue"
    else:
        color_name = "None"
    # วาดกรอบสีเทาบางๆ บน ROI
    cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (128, 128, 128), 2)

    # แสดงชื่อสีที่มีค่าสูงสุดบนมุมขวา
    cv2.putText(
        frame,
        f"{color_name}",
        (frame.shape[1] - 150, frame.shape[0] - 30),
        font,
        0.7,
        (red, blue, green),
        2,
    )

    # แสดงผลลัพธ์
    cv2.imshow("Camera", frame)
    # รอปุ่ม ESC เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ปิดกล้อง
cap.release()

# ปิดหน้าต่างผลลัพธ์
cv2.destroyAllWindows()
