import cv2
import numpy as np

class Display():

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.cap = cv2.VideoCapture(0)

        self.hand_cascade = cv2.CascadeClassifier('../dataset/cascade/hand3.xml')

    def hand_detect(self, img, gray):
        blur = cv2.GaussianBlur(gray,(5,5),0)
        hand = self.hand_cascade.detectMultiScale(blur, 1.3, 5)
        for (x, y, w, h) in hand:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.W, self.H))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, blackFrame) = cv2.threshold(grayFrame, 127, 255, cv2.THRESH_BINARY)
        self.hand_detect(frame, grayFrame)
        cv2.imshow('Show', frame)

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

