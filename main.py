from helpers.imageFilter import find_contour_center_coordinates, HSVFilter
from helpers.joint import Hand
from helpers.videoConnector import VideoReader
import cv2

reader = VideoReader("Green_Hand_Long.mp4")
hsv_filter = HSVFilter()
hand = Hand()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", hand.opencv_click)

while True:
    _, frame = reader.get_frame()
    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("frame", frame)
    key = cv2.waitKey(10)

    mask = hsv_filter.create_filtered_image(frame)

    points_mixed = find_contour_center_coordinates(mask)

    hand.update_joints_position(points_mixed)

    hand.render(frame)

    if key == ord("s"):
        hand.add_joint()
