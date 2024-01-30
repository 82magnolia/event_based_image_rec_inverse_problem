import os
import cv2
import numpy as np
import pandas as pd
import torch
import math


def load_events_and_flow(dataset):
    dataset_path = os.path.join("sample_dataset", dataset)
    events_np = np.load(os.path.join(dataset_path, "events.npy"))
    flow_np = np.load(os.path.join(dataset_path, "flow.npy"))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    events_torch = torch.from_numpy(events_np).unsqueeze(0).to(device)
    flow_torch = torch.from_numpy(flow_np).to(device)
    return events_torch, flow_torch

def load_iwe_and_flow(dataset):
    dataset_path = os.path.join("sample_dataset", dataset)
    iwe_np = np.load(os.path.join(dataset_path, "iwe.npy"))
    flow_np = np.load(os.path.join(dataset_path, "flow.npy"))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    iwe_torch = torch.from_numpy(iwe_np).to(device)
    flow_torch = torch.from_numpy(flow_np).to(device)
    return iwe_torch, flow_torch

class IJRRDataloader(object):
    def __init__(self, dataset_path):
        self.height = 180
        self.width = 240
        self.dataset_path = dataset_path
        self.calib_np = self.load_calibration(dataset_path)

        # Load events
        self.events_np = np.load(f'{self.dataset_path}/events.npz')['events']

        # Load IMU
        if os.path.exists(f'{self.dataset_path}/imu.txt'):
            imu_data = pd.read_csv(f'{self.dataset_path}/imu.txt',
                                sep=' ', header=None, engine='python')
            imu_data.columns = ["ts", "ax", "ay", "az", "gx", "gy", "gz"]
            self.imu_np = imu_data.to_numpy()
        else:
            self.imu_np = None

    def load_calibration(self, dataset_path):
        calib = np.loadtxt(f'{dataset_path}/calib.txt')
        return calib

    def load_events_and_flow(self, start_time, num_events):
        assert start_time > 0
        events_np = self.load_events(start_time, num_events)
        middle_ev_t = events_np[num_events // 2, 0]
        ev_duration = events_np[-1, 0] - events_np[0, 0]
        if self.imu_np is not None:
            imu_np = self.load_imu(start_time)
            angular_velocity = self.get_ang_vel_from_imu(imu_np, middle_ev_t)
        else:
            angular_velocity = np.zeros((3, ))
        flow_np = self.compute_flow(angular_velocity, ev_duration)
        events_torch = torch.from_numpy(events_np).unsqueeze(0)
        flow_torch = torch.from_numpy(flow_np).unsqueeze(0)
        return events_torch, flow_torch

    def load_events(self, start_time, num_events):
        # Select a slice of events
        events_t = self.events_np[:, 0]
        total_num_events = len(self.events_np)
        idx = np.searchsorted(events_t, start_time)
        if idx + num_events> total_num_events:
            events_np = np.copy(self.events_np[-num_events:, :])
        else:
            events_np = np.copy(self.events_np[idx:idx+num_events, :])
        # Get calibration data
        fx, fy, px, py = self.calib_np[0:4]
        dist_co = self.calib_np[4:]
        instrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        # Undistort events
        LUT = np.zeros([self.width, self.height, 2])
        for i in range(self.width):
            for j in range(self.height):
                LUT[i][j] = np.array([i, j])
        LUT = LUT.reshape((-1, 1, 2))
        LUT = self.undo_distortion(LUT, instrinsic_matrix, dist_co, self.width, self.height).reshape((self.width, self.height, 2))
        events_np[:, 1:3] = LUT[(events_np[:, 1]).astype(np.uint8), (events_np[:, 2]).astype(np.uint8), :]
        events_mask = (events_np[:, 1]>=0) * (events_np[:, 1]<=self.width-1) * (events_np[:, 2]>=0) * (events_np[:, 2]<=self.height-1)
        events_np = events_np[events_mask]
        return events_np
    
    def undo_distortion(self, src, instrinsic_matrix, distco, width, height):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(instrinsic_matrix, distco, (width, height), 0, (width, height))
        dst = cv2.undistortPoints(src, instrinsic_matrix, distco, None, newcameramtx)
        return dst
    
    def load_imu(self, start_time):
        imu_t = self.imu_np[:, 0]
        slice_imu_np = self.imu_np[(math.floor(start_time) <= imu_t) & (imu_t < math.ceil(start_time))]
        return slice_imu_np

    def get_ang_vel_from_imu(self, imu, seek_time):
        imu_ts = imu[:, 0]
        idx = np.searchsorted(imu_ts, seek_time)
        if idx >= imu.shape[0]: idx = imu.shape[0] - 1
        angular_velocity = imu[idx, 4:]
        return angular_velocity

    def compute_flow(self, angular_velocity, ev_duration):
        fx, fy, px, py = self.calib_np[0:4]
        map_y, map_x = np.mgrid[0:self.height, 0:self.width]
        map_y = map_y.astype(np.float32)
        map_x = map_x.astype(np.float32)
        v_bar = map_y - py
        u_bar = map_x - px
        v_dot = np.inner(np.dstack((v_bar*v_bar/fy + fy, -u_bar*v_bar/fx, -fy*u_bar/fx)), angular_velocity)
        u_dot = np.inner(np.dstack((u_bar*v_bar/fy, -u_bar*u_bar/fx - fx,  fx*v_bar/fy)), angular_velocity)
        flow_y = v_dot * ev_duration
        flow_x = u_dot * ev_duration
        flow = np.stack((flow_x, flow_y), axis=0)
        return flow

    