import pandas as pd
import numpy as np


class CSVWriter():

    def __init__(self) -> None:

        self.folder = "csv_files"
        self.x = []
        self.y = []
        self.angles = []

    def append_data(self, x, y, angles):
        self.x.append(x)
        self.y.append(y)
        self.angles.append(angles)

    def create_data_frame(self):

        self.numpy_x = self.create_2d_numpy(self.x)
        self.numpy_y = self.create_2d_numpy(self.y)
        self.numpy_angles = self.create_2d_numpy(self.angles)

        self.final_data = np.concatenate((self.numpy_x, self.numpy_y, self.numpy_angles), axis = 1)

        self.column_names = []

        for i in range(self.numpy_angles.shape[1]):
            self.column_names.append(f"Joint_x_{i}")
        for i in range(self.numpy_angles.shape[1]):
            self.column_names.append(f"Joint_y_{i}")
        for i in range(self.numpy_angles.shape[1]):
            self.column_names.append(f"Joint_angle_{i}")

    def save_excel(self):
        data_frame = pd.DataFrame(self.final_data, columns=self.column_names)
        data_frame.to_excel(f"{self.folder}/Joints.xlsx")

    def create_2d_numpy(self, data):

        n = len(max(data, key=len))

        temp_data = [x + [-1] * (n - len(x)) for x in data]
        temp_data = np.array(temp_data)

        return temp_data
