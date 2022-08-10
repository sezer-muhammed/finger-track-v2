import numpy as np
from helpers.imageFilter import ImageManager
import cv2
from math import atan2, pi


class BasicJoint():

    def __init__(self) -> None:
        self.x = 0
        self.y = 0

    def match_next(self, detection_list):
        """AI is creating summary for match_next
        This method finds next x and y value of the joint
        Args:
            detection_list ([Nx2 numpy array]): []
        """

        x_diff = detection_list[:, 0] - self.x
        y_diff = detection_list[:, 1] - self.y

        mean_sqrt_error = np.sqrt(x_diff * x_diff + y_diff * y_diff)

        min_index = np.argmin(mean_sqrt_error)

        self.x = detection_list[min_index][0]
        self.y = detection_list[min_index][1]

    def get_coordinates(self):
        """AI is creating summary for get_coordinates
        Returns coordinates as seperate variables
        """
        return self.x, self.y


class MiddleJoint(BasicJoint):

    def __init__(self) -> None:
        super().__init__()
        self.angle = 0

    def calculate_angle(self, previous_joint, next_joint):
        """AI is creating summary for calculate_angle

        Args:
            previous_joint ([BasicJoint]): []
            next_joint ([BasicJoint]): []
        """

        point_prev = np.array([previous_joint.x, previous_joint.y])
        point_now = np.array([self.x, self.y])
        point_next = np.array([next_joint.x, next_joint.y])

        now_prev = point_prev - point_now
        now_next = point_next - point_now

        cosine_angle = np.dot(now_prev, now_next) / (np.linalg.norm(now_prev) *
                                                     np.linalg.norm(now_next))
        angle = np.arccos(cosine_angle)

        self.angle = np.degrees(angle)

    def get_coordinates_and_angle(self):
        """AI is creating summary for get_coordinates_and_angle
        returns coordinates and angle for as different variables
        """
        return self.x, self.y, self.angle


class Hand(ImageManager):

    def __init__(self) -> None:

        super().__init__()
        self.frame = np.zeros((640, 480, 3), np.uint8)
        self.joints = []

    def add_joint(self):

        new_joint = MiddleJoint()
        new_joint.x = self.mouse_x
        new_joint.y = self.mouse_y
        self.joints.append(new_joint)

    def update_joints_position(self, points_mixed):

        for joint in self.joints:
            joint.match_next(points_mixed)

    def render(self, frame):

        self.update_frame(frame)
        self.draw_joints()
        self.show_image(self.frame)

    def update_frame(self, frame):
        self.frame = frame

    def draw_joints(self):

        for joint in self.joints:
            cv2.circle(self.frame, (int(joint.x), int(joint.y)), 5,
                       (0, 255, 100), 2)

    def get_joint_coordinates_and_angles(self):

        all_x = []
        all_y = []
        all_angle = []

        basic_joints = [0, len(self.joints) - 1]

        for i, joint in enumerate(self.joints):

            if i in basic_joints:
                angle = 0
                x, y = joint.get_coordinates()
            else:
                joint.calculate_angle(self.joints[i - 1], self.joints[i + 1])
                x, y, angle = joint.get_coordinates_and_angle()

            all_x.append(x)
            all_y.append(y)
            all_angle.append(angle)

        return all_x, all_y, all_angle
