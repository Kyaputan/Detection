from ultralytics import YOLO
import cv2
import os

folder_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(folder_path, "model_openvino_model")
model = YOLO(model_path,task='detect')

def detect_snake_in_video(video_path):
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



video_path = os.path.join(folder_path, "Sit_1.mp4")
detect_snake_in_video(video_path)
