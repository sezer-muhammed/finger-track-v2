import cv2
import numpy as np


class VideoReader():

    def __init__(self, cam_device=0) -> None:

        self.cam = cv2.VideoCapture(cam_device)

        self.w = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

    def get_frame(self):

        ret, frame = self.cam.read()

        return ret, frame

    def skip_frames(self, frame_number):

        self.cam.set(cv2.CAP_PROP_POS_FRAMES,
                     cv2.get(cv2.CAP_PROP_POS_FRAMES) + frame_number)
