import cv2
import numpy as np

class Display():

    def __init__(self, video, W, H):
        self.W, self.H = W, H
        self.cap = cv2.VideoCapture(0)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.W, self.H))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, blackFrame) = cv2.threshold(grayFrame, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('Show', blackFrame)
        self.surf(blackFrame)

    def run(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret:
                self.process_frame(frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        self.cap.release()
        cv2.destroyAllWindows()

