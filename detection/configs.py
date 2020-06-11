import cv2


class Config:
    def __init__(self, font=cv2.cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness