import cv2
from threading import Thread, Lock
from queue import Queue
from ultralytics import YOLO
import os
import time

class VideoProcessor:
    def __init__(self, model_path, video_path, queue_size=128):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Initialize queues for frame processing
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue(maxsize=queue_size)
        
        # Threading control
        self.stopped = False
        self.lock = Lock()
        
    def start(self):
        # Start worker threads
        Thread(target=self.read_frames, daemon=True).start()
        Thread(target=self.process_frames, daemon=True).start()
        return self
    
    def read_frames(self):
        """Thread for reading frames from video"""
        while not self.stopped:
            if not self.input_queue.full():
                success, frame = self.cap.read()
                if not success:
                    self.stop()
                    break
                
                self.input_queue.put(frame)
            else:
                time.sleep(0.001)  # Small delay to prevent CPU overuse
                
    def process_frames(self):
        """Thread for processing frames with YOLO"""
        while not self.stopped:
            if not self.input_queue.empty() and not self.output_queue.full():
                frame = self.input_queue.get()
                
                # Process frame with YOLO
                results = self.model(frame)
                annotated_frame = results[0].plot()
                
                self.output_queue.put(annotated_frame)
            else:
                time.sleep(0.001)  # Small delay to prevent CPU overuse
                
    def read(self):
        """Read processed frames from output queue"""
        return False if self.stopped and self.output_queue.empty() else self.output_queue.get()
    
    def stop(self):
        """Stop all threads"""
        self.stopped = True
        self.cap.release()
        
def main():
    # Set up paths
    model_path = os.path.join("CodeCit", "Lab1 Detection", "ex02", "Best.pt")
    video_path = os.path.join("CodeCit", "Lab1 Detection", "ex02", "Sit_1.mp4")
    
    # Initialize and start video processor
    processor = VideoProcessor(model_path, video_path).start()
    
    # Process and display video
    while True:
        # Get processed frame
        frame = processor.read()
        if frame is False:
            break
            
        # Display frame
        cv2.imshow("YOLO Inference", frame)
        
        # Check for 'q' press to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Clean up
    processor.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()