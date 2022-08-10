import cv2
import numpy as np


class ImageManager():

    def __init__(self) -> None:

        self.window_name = f"{self.__class__.__name__}"
        cv2.namedWindow(self.window_name)
        self.__width = 640
        self.mouse_x = 0
        self.mouse_y = 0

    def show_image(self, frame):
        cv2.imshow(self.window_name, frame)

    def opencv_click(self, event, x, y, *args):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y


class HSVFilter(ImageManager):

    def __init__(self) -> None:

        super().__init__()

        self.filter_values = {}
        self.hsv_names = [
            "hue_min", "saturation_min", "value_min", "hue_max",
            "saturation_max", "value_max"
        ]

        self.kernel = np.ones((5, 5), np.uint8)

        for hsv_name in self.hsv_names:
            cv2.createTrackbar(hsv_name, self.window_name, 0, 255,
                               self.set_parameters)
        self.set_parameters("PLACE HOLDER")

    def set_parameters(self, *args):

        for hsv_name in self.hsv_names:
            val = cv2.getTrackbarPos(hsv_name, self.window_name)
            self.filter_values[hsv_name] = val

    def create_filtered_image(self, frame):

        range_min = []
        range_max = []

        for i, hsv_name in enumerate(self.hsv_names):
            if i < 3:
                range_min.append(self.filter_values[hsv_name])
            else:
                range_max.append(self.filter_values[hsv_name])

        self.mask = cv2.inRange(frame, np.array(range_min),
                                np.array(range_max))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel)

        masked_image = cv2.bitwise_and(frame, frame, mask=self.mask)

        masked_image = cv2.resize(
            masked_image,
            (self._ImageManager__width, int(self._ImageManager__width * 0.55)))

        self.show_image(masked_image)

        return self.mask


def find_contour_center_coordinates(mask):

    points = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 80:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            points.append([x, y])

    points = np.array(points)

    if len(points.shape) < 2:
        points = np.expand_dims(points, axis=0)

    return points
