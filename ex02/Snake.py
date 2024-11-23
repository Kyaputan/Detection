from ultralytics import YOLO
import cv2

YOLO_Model = 'CodeCit/Lab1 Detection/ex02/best_snake.pt'
model = YOLO(YOLO_Model)

def detect_snake_in_video(video_path):
    # เปิดวิดีโอ
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    snake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # ออกจากลูปเมื่อไม่มีเฟรมอีก

        frame_count += 1

        # ตรวจทุกๆ 5 เฟรม
        if frame_count % 5 == 0:
            results = model(frame)
            snake_found = False

            for result in results:  # results เป็นลิสต์
                for detection in result.boxes:  # เข้าถึงข้อมูลการตรวจจับในแต่ละผลลัพธ์
                    class_id = int(detection.cls)  # หมายเลขคลาส
                    if class_id == 0:  # 0 คือหมายเลขคลาสสำหรับงู
                        snake_found = True
                        break
                if snake_found:
                    break

            if snake_found:
                snake_count += 1
                if snake_count == 5:  # ถ้าพบงูครบ 5 ครั้ง
                    print("Snake found")
                    snake_count = 0  # เริ่มนับใหม่
            else:
                snake_count = 0  # เริ่มนับใหม่เมื่อไม่พบงู

        # แสดงผลเฟรมในหน้าต่าง
        cv2.imshow('Video', frame)

        # กด 'q' เพื่อออกจากการแสดงผล
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # ปิดวิดีโอ
    cv2.destroyAllWindows()  # ปิดหน้าต่างที่เปิดอยู่

# เรียกใช้ฟังก์ชัน
video_path = 'CodeCit\Lab1 Detection\ex02\Snake_Test_2.mp4'  # เปลี่ยนเป็นที่อยู่ไฟล์วิดีโอของคุณ
detect_snake_in_video(video_path)
