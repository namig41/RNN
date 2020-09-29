import cv2
import numpy as np
import sdl2
import sdl2.ext
sdl2.ext.init()

class Display():

    def __init__(self, video, W, H):
        self.W, self.H = W, H
        self.cap = cv2.VideoCapture(video)

        self.window = sdl2.ext.Window("Show", size=(self.W, self.H))
        self.window.show()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.W, self.H))
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = frame.swapaxes(0, 1)
        self.window.refresh()

    def run(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret == True:
                self.process_frame(frame)
            else:
                break
    
        self.cap.release()
        cv2.destroyAllWindows()

