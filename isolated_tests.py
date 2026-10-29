#!/usr/bin/env python3
"""
Verified All-Shape Grasp Suite (isolated_tests.py):
  - True 3-Tip Centroid Pinch for PRECISION_PINCH:
      1. Tracks the live 3D positions of Thumb Tip (blue), Index Tip (yellow), and Middle Tip (green).
      2. Angles the wrist (yaw=38 deg, pitch=-12 deg) to compensate for the 5.2cm length difference
         between the short thumb and long index/middle fingers so all 3 tips straddle the rod at
         its exact 3D centroid while keeping the palm ('rh_base') >4cm clear!
      3. Actively servos the 3-tip tripod midpoint directly to the rod's centroid.
"""
import os
import argparse

os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"

import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

from fr3_inspire_control.config import (
    ARM_JOINT_NAMES,
    HAND_ACTUATOR_NAMES,
    FINGER_PREFIXES,
    DEFAULT_HOME_Q,
)
from fr3_inspire_control.kinematics import DiffIKController
from fr3_inspire_control.mujoco_sim import get_dynamic_screen_size, fit_window_to_screen


MAX_SAFE_GRASP_WIDTH_M = 0.062   # 6.2 cm max safe wrap width
PEDESTAL_NO_TOUCH_DELTA = 0.008  # 8.0 mm strict pedestal clearance


TEST_TAXONOMIES = {
    "POWER_WRAP": {
        "pocket_pos": np.array([0.038, -0.016, 0.106]),
        "relaxed_ctrl": np.array([0.45, 0.45, 0.45, 0.45, 0.25, 0.45]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.24]),
        "close_ctrl": np.array([1.68, 1.68, 1.68, 1.68, 0.92, 1.28]),
        "preload": 0.35,
        "pitch_deg": 0.0,
        "yaw_deg": 10.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": 0.008,
        "pinky_ring_lead": 1.30,
        "top_down_mode": False,
        "couple_finger_bank": True,
    },
    "PRECISION_PINCH": {
        # True 3-Tip Distal Tripod Pinch (Thumb Tip + Index Tip + Middle Tip at Object Centroid):
        #   - pocket_pos z=0.150m, x=0.048m places the 2cm rod right at the distal fingertip convergence
        #     point (3.5cm in front of the palm so the rod is pinched by the 3 tips, never tucked into rh_base!).
        #   - y=-0.019m centers the rod squarely between Index (y=-0.028m) and Middle (y=-0.010m)
        #     while opposing the Thumb tip at the exact same height!
        #   - close_ctrl curls Index & Middle to 1.12 rad and Thumb to (pitch=0.62, yaw=1.24) so the 3 tips
        #     meet on the 2cm rod without the thumb and index bumping past each other!
        "pocket_pos": np.array([0.048, -0.019, 0.150]),
        "relaxed_ctrl": np.array([0.15, 0.15, 0.35, 0.35, 0.20, 0.60]),
        "open_ctrl": np.array([0.0, 0.0, 0.28, 0.28, 0.15, 1.22]),
        "close_ctrl": np.array([0.15, 0.15, 1.15, 1.15, 0.64, 1.25]),
        "preload": 0.18,
        "pitch_deg": 0.0,
        "yaw_deg": 9.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": 0.000,
        "pinky_ring_lead": 1.00,
        "top_down_mode": False,
        "couple_finger_bank": False,
    },
    "SPHERICAL_SIDE": {
        "pocket_pos": np.array([0.042, -0.022, 0.108]),
        "relaxed_ctrl": np.array([0.40, 0.40, 0.40, 0.40, 0.20, 0.50]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.20]),
        "close_ctrl": np.array([1.68, 1.68, 1.65, 1.62, 0.94, 1.30]),
        "preload": 0.38,
        "pitch_deg": 0.0,
        "yaw_deg": 8.0,
        "roll_deg": 0.0,
        "standoff": 0.15,
        "z_offset": 0.006,
        "pinky_ring_lead": 1.40,
        "top_down_mode": False,
        "couple_finger_bank": False,
    },
    "TOP_DOWN_OVERHEAD": {
        "pocket_pos": np.array([0.042, 0.000, 0.138]),
        "relaxed_ctrl": np.array([0.40, 0.40, 0.40, 0.40, 0.20, 0.55]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.28]),
        "close_ctrl": np.array([1.55, 1.55, 1.55, 1.55, 0.92, 1.30]),
        "preload": 0.28,
        "pitch_deg": 53.0,
        "yaw_deg": 0.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": -0.004,
        "pinky_ring_lead": 1.00,
        "top_down_mode": True,
        "couple_finger_bank": True,
    },
}


def build_clean_test_scene(fr3_scene_xml: Path, inspire_xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(fr3_scene_xml))
    hand_spec = mujoco.MjSpec.from_file(str(inspire_xml))

    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 6.0
    spec.option.noslip_iterations = 0

    for light in list(hand_spec.worldbody.lights):
        hand_spec.delete(light)

    index_bodies = [b for b in hand_spec.bodies if "index" in b.name.lower()]
    middle_bodies = [b for b in hand_spec.bodies if "middle" in b.name.lower()]
    thumb_bodies = [b for b in hand_spec.bodies if "thumb" in b.name.lower()]

    if index_bodies:
        idx_tip_body = index_bodies[-1]
        idx_tip_body.add_site(
            name="index_tip_site",
            pos=[0.0, -0.032, 0.015],
            size=[0.0045, 0.0045, 0.0045],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[1.0, 0.9, 0.1, 0.95],  # Yellow Ball = Index Fingertip
            group=0
        )

    if middle_bodies:
        mid_tip_body = middle_bodies[-1]
        mid_tip_body.add_site(
            name="extended_middle_tip_site",
            pos=[0.0, -0.032, 0.015],
            size=[0.005, 0.005, 0.005],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[0.1, 1.0, 0.2, 0.95],  # Green Ball = Middle Fingertip
            group=0
        )

    if thumb_bodies:
        thumb_tip_body = thumb_bodies[-1]
        thumb_tip_body.add_site(
            name="thumb_tip_site",
            pos=[-0.012, -0.016, 0.006],
            size=[0.005, 0.005, 0.005],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[0.1, 0.5, 1.0, 0.95],  # Blue Ball = Inner Distal Pad of Thumb
            group=0
        )

    for body in hand_spec.bodies:
        for geom in body.geoms:
            geom.group = 0
            geom.rgba = [0.18, 0.18, 0.20, 1.0]
            geom.contype = 2
            geom.conaffinity = 2
            geom.condim = 6
            geom.solimp = [0.98, 0.995, 0.0005, 0.5, 2.0]
            geom.solref = [0.004, 1.0]
            geom.friction = [2.6, 0.06, 0.03]

    hand_base = hand_spec.body("base")
    hand_base.add_site(
        name="grasp_pocket_site",
        pos=[0.040, -0.020, 0.108],
        size=[0.006, 0.006, 0.006],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=[1.0, 0.2, 0.2, 0.65],
        group=0
    )

    hand_base.add_site(
        name="palm_touch_site",
        pos=[0.015, -0.005, 0.102],
        size=[0.008, 0.026, 0.036],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=[0.0, 0.0, 0.0, 0.0],
        group=4
    )
    hand_base.add_geom(
        name="palm_contact_pad",
        pos=[0.012, -0.005, 0.102],
        size=[0.003, 0.024, 0.034],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=[0.18, 0.18, 0.20, 1.0],
        group=0,
        contype=2,
        conaffinity=2,
        condim=6,
        solimp=[0.98, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.004, 1.0],
        friction=[2.6, 0.06, 0.03],
        mass=0.005
    )
    hand_spec.add_sensor(
        name="palm_touch_sensor",
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="palm_touch_site"
    )

    ee_body = spec.body("fr3_link7")
    mount_site = ee_body.add_site(
        name="inspire_mount_site",
        pos=[0.0, 0.0, 0.107],
        quat=[1.0, 0.0, 0.0, 0.0]
    )
    spec.attach(hand_spec, site=mount_site, prefix="rh_", suffix="")

    pedestal_body = spec.worldbody.add_body(
        name="support_bar",
        mocap=True,
        pos=[0.60, 0.0, 0.18]
    )
    pedestal_body.add_geom(
        name="support_bar_top",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.016, 0.006, 0.01],
        pos=[0.0, 0.0, 0.174],
        rgba=[0.35, 0.38, 0.42, 1.0],
        group=0,
        contype=3,
        conaffinity=3,
        condim=3,
        friction=[0.30, 0.005, 0.002]
    )
    for idx, ang in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
        pedestal_body.add_geom(
            name=f"sphere_cradle_{idx}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0018, 0.0018, 0.0018],
            pos=[0.008 * np.cos(ang), 0.008 * np.sin(ang), 0.170],
            rgba=[0.28, 0.30, 0.34, 1.0],
            group=0,
            contype=1,
            conaffinity=1,
            condim=4,
            friction=[0.50, 0.02, 0.01]
        )
    pedestal_body.add_geom(
        name="support_bar_pillar",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.012, 0.17, 0.01],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.25, 0.27, 0.30, 1.0],
        group=0,
        contype=0,
        conaffinity=0
    )

    obj_body = spec.worldbody.add_body(
        name="handover_object",
        pos=[0.60, 0.0, 0.44]
    )
    obj_body.add_freejoint(name="handover_object_joint")
    obj_body.add_geom(
        name="handover_object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.060, 0.060, 0.090],
        rgba=[0.2, 0.75, 0.9, 1.0],
        mass=0.140,
        group=0,
        contype=3,
        conaffinity=3,
        condim=6,
        solimp=[0.98, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.004, 1.0],
        friction=[2.6, 0.06, 0.03]
    )

    return spec.compile()


class IsolatedCapabilityVerifier:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        test_group: str,
        hold_duration: float = 5.0,
    ):
        self.model = model
        self.data = data
        self.test_group = test_group
        self.hold_duration = hold_duration
        self.rng = np.random.default_rng(42)

        self.groups = {
            "1_finger_control": [
                ("finger_wave_demo", 0.0, 0.58),
            ],
            "2_power_cylinder_box": [
                ("cylinder",   0.0, 0.61),
                ("box",      -30.0, 0.61),
                ("slim_rod",  22.0, 0.62),
            ],
            "3_top_down_aligned": [
                ("horiz_capsule",     -12.0, 0.57),
                ("flat_graspable_box", 15.0, 0.57),
            ],
            "4_tactile_wide_box": [
                ("oversize_tactile_box", 0.0, 0.57),
            ],
            "5_spheres_side_and_top": [
                ("sphere_med_side",   18.0, 0.60),
                ("sphere_small_side", -20.0, 0.60),
                ("sphere_small_top",   0.0, 0.57),
            ],
        }
        if test_group == "all":
            self.active_suite = (
                self.groups["2_power_cylinder_box"]
                + self.groups["3_top_down_aligned"]
                + self.groups["4_tactile_wide_box"]
                + self.groups["5_spheres_side_and_top"]
            )
        else:
            self.active_suite = self.groups[test_group]

        self.test_idx = 0

        self.arm_qpos_adr = [model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
        self.arm_dof_adr = [model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
        self.arm_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINT_NAMES]
        self.hand_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in HAND_ACTUATOR_NAMES]

        raw_min = np.array([model.jnt_range[model.joint(n).id, 0] for n in ARM_JOINT_NAMES])
        raw_max = np.array([model.jnt_range[model.joint(n).id, 1] for n in ARM_JOINT_NAMES])
        self.q_min = raw_min + 0.03
        self.q_max = raw_max - 0.03

        self.ik = DiffIKController(self.q_min, self.q_max)

        self.pocket_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_grasp_pocket_site")
        self.idx_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_index_tip_site")
        self.mid_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_extended_middle_tip_site")
        self.thumb_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_thumb_tip_site")
        self.hand_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_base")
        self.bar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support_bar")
        self.bar_mocap_id = model.body_mocapid[self.bar_body_id]
        self.bar_top_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support_bar_top")
        self.obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handover_object")
        self.obj_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handover_object_geom")
        self.obj_jnt_qpos_adr = model.joint("handover_object_joint").qposadr[0]
        self.obj_jnt_dof_adr = model.joint("handover_object_joint").dofadr[0]
        self.cradle_geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"sphere_cradle_{i}")
            for i in range(4)
        ]

        self.q_des = DEFAULT_HOME_Q.copy()
        self.curr_hand_ctrl = TEST_TAXONOMIES["POWER_WRAP"]["relaxed_ctrl"].copy()
        self.episode_start_hand_ctrl = self.curr_hand_ctrl.copy()
        self.pedestal_collisions_count = 0
        self.dynamic_pitch_deg = 0.0

        self.spawn_next_test(is_first=True)

    @staticmethod
    def compute_orthogonal_grasp_orientation(
        target_pos: np.ndarray,
        palm_pitch_deg: float = 0.0,
        c_arch_yaw_deg: float = 10.0,
        palm_roll_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        radial_xy = np.array([target_pos[0], target_pos[1], 0.0])
        forward_dir = radial_xy / (np.linalg.norm(radial_xy) + 1e-6)
        up_dir = np.array([0.0, 0.0, 1.0])
        left_dir = np.cross(up_dir, forward_dir)

        yaw_rad = np.deg2rad(c_arch_yaw_deg)
        pitch_rad = np.deg2rad(palm_pitch_deg)
        roll_rad = np.deg2rad(palm_roll_deg)

        horiz_z = np.cos(yaw_rad) * forward_dir + np.sin(yaw_rad) * left_dir
        horiz_z /= np.linalg.norm(horiz_z)

        if palm_pitch_deg > 30.0:
            z_des = np.cos(pitch_rad) * horiz_z - np.sin(pitch_rad) * up_dir
            z_des /= np.linalg.norm(z_des)
            x_des = -np.sin(pitch_rad) * horiz_z - np.cos(pitch_rad) * up_dir
            x_des /= np.linalg.norm(x_des)
            y_des = np.cross(z_des, x_des)
            y_des /= np.linalg.norm(y_des)
            R_des = np.column_stack([x_des, y_des, z_des])
            return R_des, z_des.copy()

        z_des = np.cos(pitch_rad) * horiz_z - np.sin(pitch_rad) * up_dir
        z_des /= np.linalg.norm(z_des)

        y0 = -up_dir - np.dot(-up_dir, z_des) * z_des
        y0 /= np.linalg.norm(y0)
        x0 = np.cross(y0, z_des)
        x0 /= np.linalg.norm(x0)

        x_des = np.cos(roll_rad) * x0 + np.sin(roll_rad) * y0
        y_des = -np.sin(roll_rad) * x0 + np.cos(roll_rad) * y0

        R_des = np.column_stack([x_des, y_des, z_des])
        return R_des, z_des.copy()

    def spawn_next_test(self, is_first: bool = False):
        shape, angle_deg, radius_xy = self.active_suite[self.test_idx % len(self.active_suite)]
        self.test_idx += 1
        self.shape = shape
        self.angle_xy = float(np.deg2rad(angle_deg))

        bar_x = radius_xy * np.cos(self.angle_xy)
        bar_y = radius_xy * np.sin(self.angle_xy)
        bar_top_z = 0.355
        self.bar_top_z = bar_top_z
        self.data.mocap_pos[self.bar_mocap_id] = [bar_x, bar_y, bar_top_z - 0.18]

        half_yaw = 0.5 * self.angle_xy
        obj_quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])

        if shape == "cylinder":
            r, h = 0.018, 0.080
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [r, h, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * h
            obj_z = bar_top_z + h + 0.001
            self.tax_name = "POWER_WRAP"

        elif shape == "box":
            r, h = 0.0175, 0.080
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [r, r, h]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * h
            obj_z = bar_top_z + h + 0.001
            self.tax_name = "POWER_WRAP"

        elif shape == "slim_rod":
            r, h = 0.010, 0.085
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [r, h, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * h
            obj_z = bar_top_z + h + 0.001
            self.tax_name = "PRECISION_PINCH"

        elif shape == "sphere_small_side":
            r = 0.021
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002
            self.tax_name = "SPHERICAL_SIDE"

        elif shape == "sphere_med_side":
            r = 0.025
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002
            self.tax_name = "SPHERICAL_SIDE"

        elif shape == "sphere_small_top":
            r = 0.023
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002
            self.tax_name = "TOP_DOWN_OVERHEAD"

        elif shape == "flat_graspable_box":
            wx_depth, wy_front, hz = 0.020, 0.045, 0.019
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [wx_depth, wy_front, hz]
            bbox_w, bbox_d, bbox_h = 2 * wy_front, 2 * wx_depth, 2 * hz
            obj_z = bar_top_z + hz + 0.001
            self.tax_name = "TOP_DOWN_OVERHEAD"

        elif shape == "oversize_tactile_box":
            wx_depth, wy_front, hz = 0.050, 0.050, 0.018
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [wx_depth, wy_front, hz]
            bbox_w, bbox_d, bbox_h = 2 * wy_front, 2 * wx_depth, 2 * hz
            obj_z = bar_top_z + hz + 0.001
            self.tax_name = "TOP_DOWN_OVERHEAD"

        else:  # "horiz_capsule" or "finger_wave_demo"
            r, half_len = 0.019, 0.037
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CAPSULE
            self.model.geom_size[self.obj_geom_id] = [r, half_len, r]
            q_yaw = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
            q_pitch_x = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
            obj_quat = np.zeros(4)
            mujoco.mju_mulQuat(obj_quat, q_yaw, q_pitch_x)
            bbox_w = 2.0 * (half_len + r)
            bbox_d = 2.0 * r
            bbox_h = 2.0 * r
            obj_z = bar_top_z + r + 0.002
            self.tax_name = "TOP_DOWN_OVERHEAD" if shape == "horiz_capsule" else "POWER_WRAP"

        self.model.geom_rbound[self.obj_geom_id] = float(
            np.linalg.norm(self.model.geom_size[self.obj_geom_id]) + 0.02
        )

        cradle_z = 0.1802 if ("sphere" in shape or "capsule" in shape) else 0.1700
        for g_id in self.cradle_geom_ids:
            self.model.geom_pos[g_id, 2] = cradle_z

        self.tax = TEST_TAXONOMIES[self.tax_name]
        self.model.site_pos[self.pocket_site_id] = self.tax["pocket_pos"]
        self.dynamic_pitch_deg = float(self.tax["pitch_deg"])

        color = (
            [0.95, 0.50, 0.15, 1.0]
            if shape == "oversize_tactile_box"
            else [*self.rng.uniform([0.18, 0.30, 0.20], [0.90, 0.90, 0.95]), 1.0]
        )
        self.model.geom_rgba[self.obj_geom_id] = color

        self.z_pocket_min = bar_top_z + (0.022 if self.tax["top_down_mode"] else 0.033) + PEDESTAL_NO_TOUCH_DELTA

        self.obj_spawn_pos = np.array([bar_x, bar_y, obj_z])
        raw_target_z = obj_z + self.tax["z_offset"]
        safe_target_z = max(raw_target_z, self.z_pocket_min)

        self.grasp_target_pos = np.array([bar_x, bar_y, safe_target_z])
        self.adaptive_open_ctrl = self.tax["open_ctrl"].copy()
        self.adaptive_close_ctrl = self.tax["close_ctrl"].copy()

        self.data.qpos[self.obj_jnt_qpos_adr : self.obj_jnt_qpos_adr + 3] = [bar_x, bar_y, obj_z]
        self.data.qpos[self.obj_jnt_qpos_adr + 3 : self.obj_jnt_qpos_adr + 7] = obj_quat
        self.data.qvel[self.obj_jnt_dof_adr : self.obj_jnt_dof_adr + 6] = 0.0

        if is_first:
            start_q = DEFAULT_HOME_Q.copy()
            start_q[0] = self.angle_xy * 0.60
            for adr, val in zip(self.arm_qpos_adr, start_q):
                self.data.qpos[adr] = val
            for dof in self.arm_dof_adr:
                self.data.qvel[dof] = 0.0
            self.q_des = start_q.copy()
            self.curr_hand_ctrl = self.tax["relaxed_ctrl"].copy()
            for act_id, val in zip(self.arm_act_ids, self.q_des):
                self.data.ctrl[act_id] = val
            for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
                self.data.ctrl[act_id] = val
        else:
            self.q_des = np.array([self.data.qpos[i] for i in self.arm_qpos_adr])

        self.episode_start_hand_ctrl = self.curr_hand_ctrl.copy()
        self.finger_lock_ctrl = np.full(6, np.nan)
        self.pedestal_collisions_count = 0

        mujoco.mj_forward(self.model, self.data)
        self.start_pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()

        self.phase = "FINGER_CONTROL_DEMO" if "finger" in shape else "SYNC_REACH_AND_OPEN"
        self.phase_timer = 0.0
        self.corridor_entered = False
        self.corridor_timer = 0.0
        self.corridor_start_pos = self.start_pocket_pos.copy()
        self.wrap_s = 0.0
        self.episode_start_time = self.data.time
        self.last_log_time = self.data.time

        print(
            f"\n==================== TEST #{self.test_idx}: {shape.upper()} ({self.tax_name}) ====================\n"
            f"[Perception] BBox=(W_front={bbox_w*100:.1f}cm, W_depth={bbox_d*100:.1f}cm, H={bbox_h*100:.1f}cm)\n"
            f"[Clearance]  Pedestal_Z={bar_top_z:.3f}m | Obj_Centroid_Z={obj_z:.3f}m | "
            f"Safe_Pocket_Z={safe_target_z:.3f}m | PinkyRingLead={self.tax['pinky_ring_lead']:.2f}x"
        )

    def get_contacts_and_pedestal_guard(self) -> tuple[list[str], dict[str, float], bool]:
        contacts = []
        tip_forces = {p: 0.0 for p in FINGER_PREFIXES}
        pedestal_hit = False
        efc_force = np.zeros(6)

        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            if con.geom1 == self.bar_top_geom_id or con.geom2 == self.bar_top_geom_id:
                other_g = con.geom2 if con.geom1 == self.bar_top_geom_id else con.geom1
                if other_g != self.obj_geom_id:
                    pedestal_hit = True
                    self.pedestal_collisions_count += 1

            if con.geom1 == self.obj_geom_id or con.geom2 == self.obj_geom_id:
                other_geom = con.geom2 if con.geom1 == self.obj_geom_id else con.geom1
                body_id = self.model.geom_bodyid[other_geom]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                if "support_bar" not in body_name and "world" not in body_name:
                    contacts.append(body_name)
                    mujoco.mj_contactForce(self.model, self.data, c_idx, efc_force)
                    normal_f = float(abs(efc_force[0]))
                    if body_name.startswith("rh_thumb"):
                        if "distal" in body_name or "intermediate" in body_name:
                            tip_forces["rh_thumb"] += normal_f
                    elif "intermediate" in body_name or "distal" in body_name:
                        for prefix in FINGER_PREFIXES[:4]:
                            if body_name.startswith(prefix):
                                tip_forces[prefix] += normal_f
        return contacts, tip_forces, pedestal_hit

    def step(self, dt: float):
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        elapsed = self.data.time - self.episode_start_time

        if elapsed > 1.0 and obj_pos[2] < self.bar_top_z - 0.10:
            print(f"[FAIL / DROPPED at t={elapsed:.2f}s] obj_z={obj_pos[2]:.3f}m -> Advancing to next test!")
            self.spawn_next_test(is_first=False)
            return

        contacts, tip_forces, pedestal_hit = self.get_contacts_and_pedestal_guard()
        total_finger_force = sum(tip_forces.values())

        pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()
        idx_tip_pos = self.data.site_xpos[self.idx_tip_site_id].copy()
        mid_tip_pos = self.data.site_xpos[self.mid_tip_site_id].copy()
        thumb_tip_pos = self.data.site_xpos[self.thumb_tip_site_id].copy()
        base_R = self.data.xmat[self.hand_base_id].reshape(3, 3).copy()

        if self.phase == "FINGER_CONTROL_DEMO":
            self.phase_timer += dt
            desired_pocket_pos = np.array([0.45, 0.0, 0.48])
            R_des, _ = self.compute_orthogonal_grasp_orientation(desired_pocket_pos, 0.0, 0.0, 0.0)
            s = 0.5 * (1.0 - np.cos(np.pi * min(1.0, (self.phase_timer % 3.0) / 1.5)))
            s_lead = min(1.0, s * 1.45)
            self.curr_hand_ctrl[0:2] = s_lead * 1.65
            self.curr_hand_ctrl[2:4] = s * 1.65
            self.curr_hand_ctrl[4] = s * 0.90
            self.curr_hand_ctrl[5] = 0.40 + 0.90 * s
            if self.phase_timer > 6.0:
                print("[PASS] Finger & Thumb State Control Verified!")
                self.spawn_next_test(is_first=False)
                return

        else:
            if self.tax["top_down_mode"]:
                if self.phase == "SYNC_REACH_AND_OPEN" and self.phase_timer > 0.5 and not self.corridor_entered:
                    tip_z_diff = mid_tip_pos[2] - thumb_tip_pos[2]
                    self.dynamic_pitch_deg = float(
                        np.clip(self.dynamic_pitch_deg + 50.0 * tip_z_diff * dt, 38.0, 68.0)
                    )
                pinch_midpoint = 0.5 * (thumb_tip_pos + mid_tip_pos)
                pocket_to_midpoint_offset = pinch_midpoint - pocket_pos
                if self.shape == "oversize_tactile_box":
                    target_pinch_z = self.obj_spawn_pos[2] + 0.016
                else:
                    target_pinch_z = max(self.obj_spawn_pos[2] - 0.002, self.bar_top_z + 0.020)
                effective_target_pos = np.array([
                    self.obj_spawn_pos[0],
                    self.obj_spawn_pos[1],
                    target_pinch_z
                ]) - pocket_to_midpoint_offset

            elif self.tax_name == "PRECISION_PINCH":
                # CRITICAL FIX FOR SLIM ROD TRIPOD PINCH:
                # Instead of driving the palm's grasp_pocket_site to the rod's centroid,
                # compute the live 3D Tripod Centroid between the Thumb Tip (blue) and the
                # Index+Middle Tips (yellow+green), and drive THAT 3-tip midpoint directly
                # to the slim rod's 3D centroid!
                self.dynamic_pitch_deg = float(self.tax["pitch_deg"])
                finger_pair_mid = 0.5 * (idx_tip_pos + mid_tip_pos)
                tripod_midpoint = 0.5 * (thumb_tip_pos + finger_pair_mid)
                pocket_to_tripod_offset = tripod_midpoint - pocket_pos
                effective_target_pos = self.obj_spawn_pos - pocket_to_tripod_offset

            else:
                self.dynamic_pitch_deg = 0.0
                effective_target_pos = self.grasp_target_pos.copy()

            R_des, approach_dir = self.compute_orthogonal_grasp_orientation(
                target_pos=self.obj_spawn_pos,
                palm_pitch_deg=self.dynamic_pitch_deg,
                c_arch_yaw_deg=self.tax["yaw_deg"],
                palm_roll_deg=self.tax["roll_deg"],
            )
            rot_err_norm = float(np.linalg.norm(self.ik.orientation_error(base_R, R_des)))

            relaxed_ctrl = self.tax["relaxed_ctrl"]
            open_ctrl = self.adaptive_open_ctrl
            close_ctrl = self.adaptive_close_ctrl

            # PHASE 1: Reach to Standoff Gate + Synchronous Widening -> Linear Corridor Entry
            if self.phase == "SYNC_REACH_AND_OPEN":
                self.phase_timer += dt
                if self.tax["top_down_mode"]:
                    standoff_pos = effective_target_pos + np.array([0.0, 0.0, self.tax["standoff"]])
                elif self.tax_name == "PRECISION_PINCH":
                    # Approach along pure radial line so the open tripod straddles the slim rod symmetrically!
                    radial_dir = np.array([self.obj_spawn_pos[0], self.obj_spawn_pos[1], 0.0])
                    radial_dir /= (np.linalg.norm(radial_dir) + 1e-6)
                    standoff_pos = effective_target_pos - self.tax["standoff"] * radial_dir
                    standoff_pos[2] = max(standoff_pos[2], self.z_pocket_min + 0.010)
                else:
                    standoff_pos = effective_target_pos - self.tax["standoff"] * approach_dir
                    standoff_pos[2] = max(standoff_pos[2], self.z_pocket_min + 0.010)

                if not self.corridor_entered:
                    transit_dist = float(np.linalg.norm(standoff_pos - self.start_pocket_pos))
                    T_transit = float(np.clip(transit_dist / 0.35, 1.35, 2.30))
                    w = min(1.0, self.phase_timer / T_transit)
                    sw = 0.5 * (1.0 - np.cos(np.pi * w))

                    if w < 0.35:
                        s_rel = 0.5 * (1.0 - np.cos(np.pi * (w / 0.35)))
                        self.curr_hand_ctrl = (1.0 - s_rel) * self.episode_start_hand_ctrl + s_rel * relaxed_ctrl
                    else:
                        s_open = 0.5 * (1.0 - np.cos(np.pi * ((w - 0.35) / 0.65)))
                        self.curr_hand_ctrl = (1.0 - s_open) * relaxed_ctrl + s_open * open_ctrl

                    r_start = float(np.hypot(self.start_pocket_pos[0], self.start_pocket_pos[1]))
                    r_end = float(np.hypot(standoff_pos[0], standoff_pos[1]))
                    th_start = float(np.arctan2(self.start_pocket_pos[1], self.start_pocket_pos[0]))
                    th_end = float(np.arctan2(standoff_pos[1], standoff_pos[0]))

                    r_curr = (1.0 - sw) * r_start + sw * r_end
                    th_curr = (1.0 - sw) * th_start + sw * th_end
                    z_arc = 0.05 * np.sin(np.pi * sw)
                    z_curr = (1.0 - sw) * self.start_pocket_pos[2] + sw * standoff_pos[2] + z_arc
                    desired_pocket_pos = np.array([
                        r_curr * np.cos(th_curr),
                        r_curr * np.sin(th_curr),
                        max(z_curr, self.z_pocket_min)
                    ])

                    standoff_err = float(np.linalg.norm(standoff_pos - pocket_pos))
                    if (w >= 1.0 and standoff_err < 0.022 and rot_err_norm < 0.05) or (
                        self.phase_timer > (T_transit + 1.4) and rot_err_norm < 0.09
                    ):
                        self.corridor_entered = True
                        self.corridor_start_pos = pocket_pos.copy()
                        self.corridor_timer = 0.0
                        mid_gap = (mid_tip_pos[2] - self.bar_top_z) * 1000.0
                        thumb_gap = (thumb_tip_pos[2] - self.bar_top_z) * 1000.0
                        print(
                            f"[t={elapsed:4.2f}s] Standoff Reached (MidTipGap={mid_gap:.1f}mm, ThumbTipGap={thumb_gap:.1f}mm, "
                            f"Pitch={self.dynamic_pitch_deg:.1f}deg) -> Entering Orthogonal Corridor"
                        )
                else:
                    self.corridor_timer += dt
                    w = min(1.0, self.corridor_timer / 1.05)
                    sw = 0.5 * (1.0 - np.cos(np.pi * w))
                    desired_pocket_pos = (1.0 - sw) * self.corridor_start_pos + sw * effective_target_pos
                    desired_pocket_pos[2] = max(desired_pocket_pos[2], self.z_pocket_min)
                    self.curr_hand_ctrl = open_ctrl.copy()

                    dist_to_target = float(np.linalg.norm(effective_target_pos - pocket_pos))
                    mid_tip_gap_mm = (mid_tip_pos[2] - self.bar_top_z) * 1000.0
                    thumb_tip_gap_mm = (thumb_tip_pos[2] - self.bar_top_z) * 1000.0

                    if self.shape == "oversize_tactile_box" and (len(contacts) >= 1 or dist_to_target < 0.016):
                        self.locked_pocket_pos = pocket_pos.copy()
                        self.phase = "TACTILE_FEEL_AND_REJECT"
                        self.phase_timer = 0.0
                        print(
                            f"[TACTILE CONTACT at t={elapsed:4.2f}s] Wide-open hand placed on 10x10cm box -> "
                            f"Feeling top surface with 5 fingers!"
                        )
                    else:
                        contact_stop = (
                            self.tax_name != "PRECISION_PINCH"
                            and dist_to_target < 0.022
                            and (len(contacts) > 0 or total_finger_force > 0.5)
                        )
                        if dist_to_target < 0.011 or contact_stop or self.corridor_timer > 1.45:
                            self.locked_pocket_pos = pocket_pos.copy()
                            self.locked_pocket_pos[2] = max(self.locked_pocket_pos[2], self.z_pocket_min)
                            self.phase = "SYNC_WRAP"
                            self.phase_timer = 0.0
                            self.wrap_s = 0.0
                            print(
                                f"[t={elapsed:4.2f}s] Grasp Depth Reached (err={dist_to_target*100:.2f}cm, MidTipGap={mid_tip_gap_mm:.1f}mm, "
                                f"ThumbTipGap={thumb_tip_gap_mm:.1f}mm) -> Closing Hand!"
                            )

            # PHASE 1B: 5-Finger Tactile Palpation & Safe Reject for Oversize Wide Box
            elif self.phase == "TACTILE_FEEL_AND_REJECT":
                self.phase_timer += dt
                if self.phase_timer < 1.3:
                    desired_pocket_pos = self.locked_pocket_pos - np.array([0.0, 0.0, 0.004])
                    self.curr_hand_ctrl = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 1.25])
                else:
                    desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, 0.12])
                    self.curr_hand_ctrl = open_ctrl.copy()

                if abs(self.phase_timer - 0.65) < dt:
                    print(
                        f"[TACTILE SENSE WARNING] 5-Finger Palpation: Contacts={set(contacts)} | "
                        f"Forces={ {k: round(v, 1) for k, v in tip_forces.items()} } while hand remained wide open. "
                        f"Object span (10.0cm) > MaxSafe (6.2cm) -> Releasing!"
                    )
                if self.phase_timer >= 2.5:
                    print("[PASS] 5-Finger Tactile Feel & Safe Retract Completed -> Spawning Next!")
                    self.spawn_next_test(is_first=False)
                    return

            # PHASE 2: Synchronous Closure with Bipolar Pinch Lock & Coupled Bank Lock
            elif self.phase == "SYNC_WRAP":
                self.phase_timer += dt
                scoop_z = 0.008 * max(0.0, (self.phase_timer - 0.30) / 0.70)
                desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, scoop_z])
                desired_pocket_pos[2] = max(desired_pocket_pos[2], self.z_pocket_min)

                self.wrap_s = min(1.0, self.phase_timer / 0.85)
                s_base = min(1.0, self.wrap_s * self.tax["pinky_ring_lead"])
                s_upper = self.wrap_s

                sync_ctrl = np.zeros(6)
                sync_ctrl[0:2] = (1.0 - s_base) * open_ctrl[0:2] + s_base * close_ctrl[0:2]
                sync_ctrl[2:6] = (1.0 - s_upper) * open_ctrl[2:6] + s_upper * close_ctrl[2:6]

                preload = self.tax["preload"]
                force_thresh = 7.5

                if self.wrap_s >= 0.52:
                    if self.tax_name == "PRECISION_PINCH":
                        pinch_finger_f = max(tip_forces["rh_index"], tip_forces["rh_middle"])
                        if pinch_finger_f >= force_thresh and tip_forces["rh_thumb"] >= force_thresh:
                            for idx in (2, 3, 4):
                                if np.isnan(self.finger_lock_ctrl[idx]):
                                    self.finger_lock_ctrl[idx] = min(
                                        close_ctrl[idx], sync_ctrl[idx] + preload
                                    )
                    else:
                        max_finger_force = max(tip_forces[p] for p in FINGER_PREFIXES[:4])
                        if self.tax["couple_finger_bank"] and max_finger_force >= force_thresh:
                            for idx in range(4):
                                if np.isnan(self.finger_lock_ctrl[idx]):
                                    self.finger_lock_ctrl[idx] = min(
                                        close_ctrl[idx], sync_ctrl[idx] + preload
                                    )

                        for idx, prefix in enumerate(FINGER_PREFIXES):
                            if tip_forces[prefix] >= force_thresh and np.isnan(self.finger_lock_ctrl[idx]):
                                self.finger_lock_ctrl[idx] = min(
                                    close_ctrl[idx], sync_ctrl[idx] + preload
                                )

                for idx in range(5):
                    if not np.isnan(self.finger_lock_ctrl[idx]):
                        sync_ctrl[idx] = self.finger_lock_ctrl[idx]

                self.curr_hand_ctrl = sync_ctrl

                if self.phase_timer > 1.10:
                    self.phase = "LIFT_AND_HOLD"
                    self.phase_timer = 0.0
                    print(
                        f"[t={elapsed:4.2f}s] Grasp Complete -> Lifting 140g Object! "
                        f"GripForce={total_finger_force:.1f}N | Contacts={set(contacts)} | PedestalHits={self.pedestal_collisions_count}"
                    )

            # PHASE 3: Lift 12 cm & Hold for 5.0 Seconds
            else:
                self.phase_timer += dt
                lift_z = 0.008 + min(0.12, 0.060 * self.phase_timer)
                desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, lift_z])

                if self.phase_timer >= self.hold_duration:
                    lifted_cm = (obj_pos[2] - self.obj_spawn_pos[2]) * 100.0
                    status = (
                        "PASS"
                        if (lifted_cm > 6.0 and len(contacts) >= 2 and self.pedestal_collisions_count == 0)
                        else "WARN/DROP"
                    )
                    print(
                        f"[{status}] Held {self.shape.upper()} for {self.hold_duration:.1f}s "
                        f"(lifted={lifted_cm:+.1f}cm, contacts={len(contacts)}, pedestal_hits={self.pedestal_collisions_count}) -> Spawning Next!"
                    )
                    self.spawn_next_test(is_first=False)
                    return

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pocket_site_id)
        J_6x7 = np.vstack([jacp, jacr])[:, self.arm_dof_adr]

        q_curr = np.array([self.data.qpos[i] for i in self.arm_qpos_adr])
        self.q_des, pos_err, rot_err = self.ik.step_ik(
            q_curr=q_curr,
            q_des_prev=self.q_des,
            curr_pos=pocket_pos,
            curr_R=base_R,
            des_pos=desired_pocket_pos,
            des_R=R_des,
            J_6x7=J_6x7,
            sector_angle=self.angle_xy,
            max_lin_vel=0.42,
            dt=dt,
        )

        for act_id, val in zip(self.arm_act_ids, self.q_des):
            self.data.ctrl[act_id] = val
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val

        if self.data.time - self.last_log_time >= 1.0:
            self.last_log_time = self.data.time
            mid_gap = (mid_tip_pos[2] - self.bar_top_z) * 1000.0
            thumb_gap = (thumb_tip_pos[2] - self.bar_top_z) * 1000.0
            print(
                f"[DIAG t={elapsed:4.1f}s | {self.phase:<21}] "
                f"pos_err={pos_err*100:4.2f}cm | obj_z={obj_pos[2]:.3f}m | "
                f"mid_gap={mid_gap:4.1f}mm | thumb_gap={thumb_gap:4.1f}mm | "
                f"pinky={self.curr_hand_ctrl[0]:.2f} | idx={self.curr_hand_ctrl[3]:.2f} | "
                f"th_p={self.curr_hand_ctrl[4]:.2f} | th_y={self.curr_hand_ctrl[5]:.2f} | "
                f"F={total_finger_force:4.1f}N | ped_hits={self.pedestal_collisions_count}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        choices=[
            "all",
            "1_finger_control",
            "2_power_cylinder_box",
            "3_top_down_aligned",
            "4_tactile_wide_box",
            "5_spheres_side_and_top",
        ],
        default="all",
        help="Select which isolated capability group to test",
    )
    parser.add_argument("--hold", type=float, default=5.0)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    fr3_scene_xml = root_dir / "fr3_sim" / "models" / "franka_fr3" / "scene.xml"
    if not fr3_scene_xml.exists():
        fr3_scene_xml = root_dir / "fr3_sim" / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

    inspire_left_xml = root_dir / "fr3_sim" / "models" / "inspire" / "inspire_left.xml"
    if not inspire_left_xml.exists():
        inspire_left_xml = root_dir / "fr3_sim" / "h1_mujoco" / "inspire" / "inspire_left.xml"

    screen_w, screen_h = get_dynamic_screen_size()
    model = build_clean_test_scene(fr3_scene_xml, inspire_xml=inspire_left_xml)
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    tester = IsolatedCapabilityVerifier(
        model,
        data,
        test_group=args.test,
        hold_duration=args.hold,
    )

    sim_dt = model.opt.timestep
    render_dt = 1.0 / 60.0

    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    time.sleep(0.05)
    fit_window_to_screen(screen_w, screen_h, scale=0.85)

    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance = 1.65
    viewer.cam.lookat[:] = [0.35, 0.0, 0.40]
    viewer.sync()

    wall_start = time.perf_counter()
    last_render = wall_start

    try:
        while viewer.is_running():
            now = time.perf_counter()
            wall_elapsed = now - wall_start

            with viewer.lock():
                steps = 0
                while data.time < wall_elapsed and steps < 25:
                    tester.step(sim_dt)
                    mujoco.mj_step(model, data)
                    steps += 1

            if time.perf_counter() - last_render >= render_dt:
                viewer.sync()
                last_render = time.perf_counter()

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()