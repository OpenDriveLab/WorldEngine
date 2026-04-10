import pickle
import numpy as np
import torch
import mmcv
from pyquaternion import Quaternion

class ScorePostProcessor:
    def __init__(
            self,
            config,
            current_pkl,
            **kwargs
            ):

        self.traj = np.load(config)
        self.data_info = mmcv.load(current_pkl, file_format="pkl")

    def process(self, result):
        plan_idx = result['chosen_ind']
        plan_traj = self.traj[plan_idx]
        # output_planning_traj = result['trajectory']     # should be the same
        return plan_traj, plan_idx

    def global_transform(self, plan_traj):
        l2g_r_mat, l2g_t = self.update_transform(self.data_info)
        traj_global = (l2g_r_mat @ plan_traj.T).T + l2g_t
        return traj_global

    def update_transform(self, info):
        l2e_r = info["lidar2ego_rotation"]  # List[float], (4, ),
        l2e_t = info["lidar2ego_translation"]  # List[float], (3, ),
        l2e_r_mat: np.array = Quaternion(l2e_r).rotation_matrix  # 3 x 3
        l2e = np.identity(4)
        l2e[:3, :3] = l2e_r_mat
        l2e[:3, 3] = l2e_t  # 4 x 4

        # ego to global
        e2g_r = info[
            "ego2global_rotation"
        ]  # List[float], (4, ) quaternion, in global coordinate
        e2g_t = info[
            "ego2global_translation"
        ]  # List[float], (3, ), in global coordinate
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        e2g = np.identity(4)
        e2g[:3, :3] = e2g_r_mat
        e2g[:3, 3] = e2g_t  # 4 x 4

        # lidar to global
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        return l2g_r_mat, l2g_t
