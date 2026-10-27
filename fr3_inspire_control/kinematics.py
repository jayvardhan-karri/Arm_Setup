#!/usr/bin/env python3
import numpy as np
from .config import DEFAULT_NULL_Q, MAX_ARM_JOINT_VEL


class DiffIKController:
    """
    Hardware-agnostic Damped Least Squares (DLS) Differential IK solver
    with SO(3) orientation alignment and active joint-limit repulsion in the nullspace.
    """
    def __init__(self, q_min: np.ndarray, q_max: np.ndarray):
        self.q_min = q_min.copy()
        self.q_max = q_max.copy()
        self.q_mid = 0.5 * (self.q_min + self.q_max)
        self.q_span = np.maximum(self.q_max - self.q_min, 1e-3)

        self.kp_pos = 8.0
        self.kp_rot = 7.0
        self.damping = 0.045
        self.null_weights = np.array([1.5, 1.8, 2.8, 3.2, 3.5, 2.8, 1.0])

    @staticmethod
    def compute_grasp_orientation(
        target_pos: np.ndarray,
        palm_pitch_deg: float = 0.0,
        c_arch_yaw_deg: float = 14.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the right-handed SO(3) rotation matrix R_des for the Inspire Left Hand
        and the unit approach vector along which the C-arch opens.
        """
        radial_xy = np.array([target_pos[0], target_pos[1], 0.0])
        forward_dir = radial_xy / (np.linalg.norm(radial_xy) + 1e-6)
        up_dir = np.array([0.0, 0.0, 1.0])
        left_dir = np.cross(up_dir, forward_dir)

        yaw_rad = np.deg2rad(c_arch_yaw_deg)
        pitch_rad = np.deg2rad(palm_pitch_deg)

        horiz_z = np.cos(yaw_rad) * forward_dir + np.sin(yaw_rad) * left_dir
        horiz_z /= np.linalg.norm(horiz_z)

        z_des = np.cos(pitch_rad) * horiz_z - np.sin(pitch_rad) * up_dir
        z_des /= np.linalg.norm(z_des)

        # Thumb side (-Y_base) points up/back perpendicular to z_des
        y_des = -up_dir - np.dot(-up_dir, z_des) * z_des
        y_des /= np.linalg.norm(y_des)

        # Inner palm normal (+X_base) guaranteed right-handed: x_des = y_des x z_des
        x_des = np.cross(y_des, z_des)
        x_des /= np.linalg.norm(x_des)

        R_des = np.column_stack([x_des, y_des, z_des])
        approach_dir = z_des.copy()
        return R_des, approach_dir

    @staticmethod
    def orientation_error(curr_R: np.ndarray, des_R: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(curr_R[:, 0], des_R[:, 0]) +
            np.cross(curr_R[:, 1], des_R[:, 1]) +
            np.cross(curr_R[:, 2], des_R[:, 2])
        )

    def step_ik(
        self,
        q_curr: np.ndarray,
        q_des_prev: np.ndarray,
        curr_pos: np.ndarray,
        curr_R: np.ndarray,
        des_pos: np.ndarray,
        des_R: np.ndarray,
        J_6x7: np.ndarray,
        sector_angle: float,
        max_lin_vel: float,
        dt: float
    ) -> tuple[np.ndarray, float, float]:
        pos_err = des_pos - curr_pos
        rot_err = self.orientation_error(curr_R, des_R)

        lin_vel = self.kp_pos * pos_err
        lin_norm = float(np.linalg.norm(lin_vel))
        if lin_norm > max_lin_vel:
            lin_vel *= (max_lin_vel / lin_norm)

        # Clamp angular velocity to 1.6 rad/s so the wrist NEVER whips/dances ahead of the arm!
        ang_vel = self.kp_rot * rot_err
        max_ang_vel = 1.6
        ang_norm = float(np.linalg.norm(ang_vel))
        if ang_norm > max_ang_vel:
            ang_vel *= (max_ang_vel / ang_norm)

        twist = np.concatenate([lin_vel, ang_vel])

        JJT = J_6x7 @ J_6x7.T + (self.damping ** 2) * np.eye(6)
        J_pinv = J_6x7.T @ np.linalg.solve(JJT, np.eye(6))

        # Smoothly track the current hand azimuth in nullspace (eliminates cross-sector elbow dance!)
        curr_azimuth = float(np.arctan2(curr_pos[1], max(curr_pos[0], 0.15)))
        null_target = DEFAULT_NULL_Q.copy()
        null_target[0] = 0.5 * (curr_azimuth + sector_angle)
        posture_grad = self.null_weights * (null_target - q_curr)

        margin = 0.20 * self.q_span
        dist_low = q_curr - self.q_min
        dist_high = self.q_max - q_curr
        limit_repulsion = np.zeros(7)
        for i in range(7):
            if dist_low[i] < margin[i]:
                limit_repulsion[i] += 6.5 * (margin[i] - dist_low[i]) / margin[i]
            if dist_high[i] < margin[i]:
                limit_repulsion[i] -= 6.5 * (margin[i] - dist_high[i]) / margin[i]

        dq_null = (np.eye(7) - J_pinv @ J_6x7) @ (posture_grad + limit_repulsion)
        dq = J_pinv @ twist + dq_null
        dq = np.clip(dq, -MAX_ARM_JOINT_VEL, MAX_ARM_JOINT_VEL)

        q_des = q_des_prev + dq * dt
        q_des = np.clip(q_des, q_curr - 0.25, q_curr + 0.25)
        q_des = np.clip(q_des, self.q_min, self.q_max)

        return q_des, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))