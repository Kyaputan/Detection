import face_recognition
import cv2
import threading
import time
from datetime import datetime
import os
import numpy as np

class FaceRecognitionSystem:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.fps = 0
        self.frame_time = 0
        self.is_recording = False
        self.video_writer = None
        self.frame_lock = threading.Lock()
        
    def load_known_faces(self, image_paths, names):
        """โหลดภาพใบหน้าที่รู้จักเข้าระบบ"""
        if len(image_paths) != len(names):
            raise ValueError("จำนวนรูปภาพและชื่อต้องเท่ากัน")
            
        for image_path, name in zip(image_paths, names):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            print(f"Loaded face: {name}")

    def process_frame(self, frame):
        """ประมวลผลเฟรมเพื่อตรวจจับใบหน้า"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

    def draw_results(self, frame):
        """วาดผลลัพธ์บนเฟรม"""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # วาดกรอบใบหน้า
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # วาดพื้นหลังสำหรับชื่อ
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        # แสดง FPS
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        
        # แสดงสถานะการบันทึก
        if self.is_recording:
            cv2.putText(frame, "Recording", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

        return frame

    def start_recording(self):
        """เริ่มบันทึกวิดีโอ"""
        if not self.is_recording:
            # สร้างโฟลเดอร์ recordings ถ้ายังไม่มี
            if not os.path.exists('recordings'):
                os.makedirs('recordings')
                
            # สร้างชื่อไฟล์จากเวลาปัจจุบัน
            filename = f"recordings/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            
            # กำหนดค่า VideoWriter
            frame_width = int(self.video_capture.get(3))
            frame_height = int(self.video_capture.get(4))
            self.video_writer = cv2.VideoWriter(filename, 
                                              cv2.VideoWriter_fourcc(*'XVID'),
                                              20.0, 
                                              (frame_width, frame_height))
            self.is_recording = True
            print(f"Started recording: {filename}")

    def stop_recording(self):
        """หยุดบันทึกวิดีโอ"""
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("Stopped recording")

    def run(self):
        """เริ่มการทำงานของระบบ"""
        prev_time = time.time()
        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                # คำนวณ FPS
                current_time = time.time()
                self.fps = 1 / (current_time - prev_time)
                prev_time = current_time

                # ประมวลผลเฟรม
                self.process_frame(frame)
                
                # วาดผลลัพธ์
                frame = self.draw_results(frame)

                # บันทึกวิดีโอ (ถ้ากำลังบันทึกอยู่)
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)

                # แสดงผล
                cv2.imshow('Face Recognition', frame)

                # รับค่าคีย์บอร์ด
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # กด 'q' เพื่อออก
                    break
                elif key == ord('r'):  # กด 'r' เพื่อเริ่ม/หยุดการบันทึก
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

        finally:
            # ทำความสะอาดและปิดโปรแกรม
            if self.is_recording:
                self.stop_recording()
            self.video_capture.release()
            cv2.destroyAllWindows()

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    
    # โหลดภาพที่รู้จัก
    image_paths = [
        "CodeCit/Lab1-Detection/ex08/tee.jpg"
    ]
    names = ["Cap"]
    
    face_system.load_known_faces(image_paths, names)
    face_system.run()