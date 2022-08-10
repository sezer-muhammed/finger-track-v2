import csv
from helpers.csvCreator import CSVWriter
from helpers.imageFilter import find_contour_center_coordinates, HSVFilter
from helpers.joint import Hand
from helpers.videoConnector import VideoReader
import cv2

reader = VideoReader("green_hand.mp4")
hsv_filter = HSVFilter()
hand = Hand()
csv_writer = CSVWriter()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", hand.opencv_click)

while True:
    _, frame = reader.get_frame()
    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("frame", frame)
    key = cv2.waitKey(100)

    mask = hsv_filter.create_filtered_image(frame)

    points_mixed = find_contour_center_coordinates(mask)

    hand.update_joints_position(points_mixed)

    hand.render(frame)

    X_j, Y_j, Angles_j = hand.get_joint_coordinates_and_angles()

    csv_writer.append_data(X_j, Y_j, Angles_j)

    if key == ord("s"):
        hand.add_joint()
    
    if key == ord("q"):
        break

csv_writer.create_data_frame()
csv_writer.save_excel()