#!/usr/bin/env python3
"""
Adaptive Perception-Driven Human Grasp Testbench:
  1. True Side Grasp for Spheres & Cylinders/Boxes (SPHERICAL_SIDE & POWER_WRAP, roll=0, pitch=0)
     with Pedestal Tolerance Stop + Faster Pinky/Ring Base-Securing Closure!
  2. True Vertical Top-Down Grasp (TOP_DOWN_OVERHEAD) with Fully Extended Hand:
     Descends straight down from +Z without bumping the object, stopping when the
     MIDDLE FINGERTIP is at height Delta (8mm) above the pedestal, straddling the object
     between the extended fingers and opposed thumb!
  3. Max Safe Grasp Width Gate (Warns & skips safely when min grasp width > 6.2 cm)
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


MAX_SAFE_GRASP_WIDTH_M = 0.062   # 6.2 cm conservative safe wrap limit for Inspire Hand
PEDESTAL_NO_TOUCH_DELTA = 0.008  # 8.0 mm strict minimum clearance above pedestal surface


TEST_TAXONOMIES = {
    "POWER_WRAP": {
        "pocket_pos": np.array([0.040, -0.020, 0.108]),
        "relaxed_ctrl": np.array([0.45, 0.45, 0.45, 0.45, 0.25, 0.45]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.22]),
        "close_ctrl": np.array([1.65, 1.65, 1.65, 1.65, 0.85, 1.26]),
        "preload": 0.25,
        "pitch_deg": 0.0,
        "yaw_deg": 8.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": 0.015,
        "pinky_ring_lead": 1.35,  # Pinky & ring close 35% faster to secure object base!
        "top_down_mode": False,
    },
    "PRECISION_PINCH": {
        "pocket_pos": np.array([0.044, -0.022, 0.136]),
        "relaxed_ctrl": np.array([0.35, 0.35, 0.40, 0.40, 0.20, 0.50]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.08]),
        "close_ctrl": np.array([0.20, 0.20, 1.58, 1.58, 0.85, 1.30]),
        "preload": 0.22,
        "pitch_deg": 0.0,
        "yaw_deg": 11.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": 0.015,
        "pinky_ring_lead": 1.0,
        "top_down_mode": False,
    },
    "SPHERICAL_SIDE": {
        # Pure Horizontal Side Grasp (pitch=0, roll=0):
        # Hand stops at Pedestal Tolerance height so lower palm/pinky/ring sit right at the sphere's
        # equator (8mm above pedestal). Pinky & Ring close 40% faster to scoop & lock the base of the
        # sphere while the opposed thumb presses from the opposite side!
        "pocket_pos": np.array([0.041, -0.010, 0.118]),
        "relaxed_ctrl": np.array([0.40, 0.40, 0.40, 0.40, 0.25, 0.55]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.24]),
        "close_ctrl": np.array([1.68, 1.68, 1.58, 1.55, 0.88, 1.30]),
        "preload": 0.32,
        "pitch_deg": 0.0,
        "yaw_deg": 9.0,
        "roll_deg": 0.0,
        "standoff": 0.14,
        "z_offset": 0.0,
        "pinky_ring_lead": 1.40,  # Fast pinky & ring closure secures bottom hemisphere of sphere!
        "top_down_mode": False,
    },
    "TOP_DOWN_OVERHEAD": {
        # True Top-Down Extended-Hand Grasp from Above (+Z):
        # Hand points downward (pitch=78 deg) with wrist rolled (roll=85 deg) so the open C-arch
        # straddles the horizontal capsule/sphere from above.
        # Stops strictly when the FULLY EXTENDED MIDDLE FINGERTIP is at height Delta (8mm) above the pedestal!
        "pocket_pos": np.array([0.042, -0.018, 0.125]),
        "relaxed_ctrl": np.array([0.40, 0.40, 0.40, 0.40, 0.20, 0.55]),
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.24]),
        "close_ctrl": np.array([1.62, 1.62, 1.62, 1.62, 0.88, 1.28]),
        "preload": 0.30,
        "pitch_deg": 78.0,
        "yaw_deg": 0.0,
        "roll_deg": 85.0,
        "standoff": 0.15,
        "z_offset": 0.0,
        "pinky_ring_lead": 1.25,
        "top_down_mode": True,
    },
}


def build_clean_test_scene(fr3_scene_xml: Path, inspire_xml: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(fr3_scene_xml))
    hand_spec = mujoco.MjSpec.from_file(str(inspire_xml))

    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.noslip_iterations = 2

    for light in list(hand_spec.worldbody.lights):
        hand_spec.delete(light)

    for body in hand_spec.bodies:
        for geom in body.geoms:
            if geom.classname.name == "visual" or (geom.contype == 0 and geom.conaffinity == 0):
                geom.group = 0
                geom.rgba = [0.18, 0.18, 0.20, 1.0]
            else:
                geom.contype = 2
                geom.conaffinity = 2
                geom.condim = 6
                geom.solimp = [0.97, 0.995, 0.0005, 0.5, 2.0]
                geom.solref = [0.005, 1.0]
                geom.friction = [2.4, 0.06, 0.03]

    hand_base = hand_spec.body("base")
    hand_base.add_site(
        name="grasp_pocket_site",
        pos=[0.040, -0.020, 0.108],
        size=[0.006, 0.006, 0.006],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=[1.0, 0.2, 0.2, 0.65],
        group=0
    )

    # Add a site right at the extended Middle Fingertip (z=0.186m in base frame) to measure exact tip clearance!
    hand_base.add_site(
        name="extended_middle_tip_site",
        pos=[0.018, -0.005, 0.186],
        size=[0.004, 0.004, 0.004],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=[0.2, 1.0, 0.2, 0.5],
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
        solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.005, 1.0],
        friction=[2.5, 0.08, 0.04],
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

    # Support Pedestal with REAL Collision (contype=3, conaffinity=3 collides with BOTH object and hand!)
    pedestal_body = spec.worldbody.add_body(
        name="support_bar",
        mocap=True,
        pos=[0.60, 0.0, 0.18]
    )
    pedestal_body.add_geom(
        name="support_bar_top",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.024, 0.006, 0.01],
        pos=[0.0, 0.0, 0.174],
        rgba=[0.35, 0.38, 0.42, 1.0],
        group=0,
        contype=3,
        conaffinity=3,
        condim=6,
        friction=[0.60, 0.03, 0.02]
    )
    for idx, ang in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
        pedestal_body.add_geom(
            name=f"sphere_cradle_{idx}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.002, 0.002, 0.002],
            pos=[0.010 * np.cos(ang), 0.010 * np.sin(ang), 0.170],
            rgba=[0.28, 0.30, 0.34, 1.0],
            group=0,
            contype=1,
            conaffinity=1,
            condim=6,
            friction=[0.85, 0.04, 0.03]
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
        size=[0.018, 0.018, 0.080],
        rgba=[0.2, 0.75, 0.9, 1.0],
        mass=0.065,
        group=0,
        contype=3,
        conaffinity=3,
        condim=6,
        solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.005, 1.0],
        friction=[2.4, 0.06, 0.03]
    )

    return spec.compile()


class AdaptivePerceptionGraspTester:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mode: str,
        shape: str,
        angle_deg: float,
        hold_duration: float = 5.0,
    ):
        self.model = model
        self.data = data
        self.mode = mode
        self.single_shape = shape
        self.single_angle_deg = angle_deg
        self.hold_duration = hold_duration
        self.rng = np.random.default_rng(42)

        # Test suite covering:
        #   1. SPHERICAL_SIDE (Pure Horizontal Side Grasp with Pinky/Ring Base-Securing Lead)
        #   2. TOP_DOWN_OVERHEAD (Vertical Descent from +Z stopping at Middle-Fingertip Tolerance Delta)
        #   3. SPHERE_OVERSIZE (Safety Gate > 6.2cm)
        #   4. POWER_WRAP & PRECISION_PINCH
        self.test_suite = [
            ("sphere_med",      20.0, 0.61),  # 5.2 cm sphere -> SPHERICAL_SIDE (Horizontal Side Grasp + Pinky/Ring Lead!)
            ("sphere_small",   -25.0, 0.60),  # 3.8 cm sphere -> SPHERICAL_SIDE (Horizontal Side Grasp + Pedestal Delta!)
            ("horiz_capsule",  -15.0, 0.56),  # 11.2 cm wide x 3.6 cm waist -> TOP_DOWN_OVERHEAD (Vertical from Up!)
            ("sphere_top",       0.0, 0.57),  # 5.0 cm sphere -> TOP_DOWN_OVERHEAD (Grabbing Sphere from Up!)
            ("sphere_oversize",  0.0, 0.61),  # 7.8 cm sphere (> 6.2cm max safe wrap) -> SAFETY WARNING & SKIP!
            ("cylinder",         0.0, 0.61),  # 3.6 cm upright cylinder -> POWER_WRAP (with Pinky/Ring Lead!)
            ("box",            -35.0, 0.61),  # 3.5 cm upright cuboid -> POWER_WRAP (with Pinky/Ring Lead!)
            ("slim_rod",        25.0, 0.63),  # 2.0 cm slim rod -> PRECISION_PINCH
        ]
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
        self.mid_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rh_extended_middle_tip_site")
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

        self.spawn_next_test(is_first=True)

    @staticmethod
    def compute_grasp_orientation_with_roll(
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

    @staticmethod
    def compute_lowest_hand_point_offset(pitch_deg: float, roll_deg: float, pocket_z_local: float) -> float:
        """
        Computes exact vertical drop from grasp_pocket_site to the lowest point of the extended hand:
          - For horizontal side grasps (pitch=0, roll=0): lowest point is the pinky edge of the hand
            (dy = +0.026m - (-0.010m) = 0.036m below pocket_site).
          - For top-down overhead grasps (pitch=78 deg): lowest point is the FULLY EXTENDED MIDDLE FINGERTIP
            (z_tip_local = 0.186m vs pocket_z_local = 0.125m -> dz = 0.061m * sin(pitch) below pocket_site!).
        """
        pitch_rad = np.deg2rad(pitch_deg)
        roll_rad = np.deg2rad(roll_deg)

        # 1. Extended middle fingertip vertical drop below pocket_site
        dz_tip = max(0.0, 0.186 - pocket_z_local)
        middle_tip_drop = dz_tip * np.sin(pitch_rad) + 0.012 * np.cos(pitch_rad)

        # 2. Pinky edge vertical drop below pocket_site (for horizontal side grasp)
        pinky_edge_drop = 0.035 * np.cos(roll_rad) * np.cos(pitch_rad)

        return float(max(middle_tip_drop, pinky_edge_drop))

    def select_grasp_normal_and_taxonomy(
        self, shape: str, bbox_w_front: float, bbox_depth: float, bbox_height: float
    ) -> tuple[str, float, bool]:
        if shape == "sphere_top":
            return "TOP_DOWN_OVERHEAD", bbox_depth, (bbox_depth <= MAX_SAFE_GRASP_WIDTH_M)

        if "sphere" in shape:
            grasp_width = bbox_w_front
            is_safe = (grasp_width <= MAX_SAFE_GRASP_WIDTH_M)
            return "SPHERICAL_SIDE", grasp_width, is_safe

        # Least-width normal check: if frontal width > 6.2cm but top-down waist <= 6.2cm, grab from UP!
        if bbox_w_front > MAX_SAFE_GRASP_WIDTH_M and bbox_depth <= MAX_SAFE_GRASP_WIDTH_M:
            return "TOP_DOWN_OVERHEAD", bbox_depth, True

        grasp_width = bbox_w_front
        is_safe = (grasp_width <= MAX_SAFE_GRASP_WIDTH_M)
        if grasp_width <= 0.024:
            return "PRECISION_PINCH", grasp_width, is_safe
        return "POWER_WRAP", grasp_width, is_safe

    def spawn_next_test(self, is_first: bool = False):
        if self.mode == "suite":
            shape, angle_deg, radius_xy = self.test_suite[self.test_idx % len(self.test_suite)]
        else:
            shape = self.single_shape
            angle_deg = self.single_angle_deg
            radius_xy = 0.58 if shape in ("horiz_capsule", "sphere_top") else 0.60

        self.test_idx += 1
        self.shape = shape
        self.angle_xy = float(np.deg2rad(angle_deg))

        bar_x = radius_xy * np.cos(self.angle_xy)
        bar_y = radius_xy * np.sin(self.angle_xy)
        bar_top_z = float(self.rng.uniform(0.34, 0.37))
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

        elif shape == "box":
            r, h = 0.0175, 0.080
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [r, r, h]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * h
            obj_z = bar_top_z + h + 0.001

        elif shape == "slim_rod":
            r, h = 0.010, 0.085
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [r, h, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * h
            obj_z = bar_top_z + h + 0.001

        elif shape == "sphere_small":
            r = 0.020  # 4.0 cm sphere
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002

        elif shape in ("sphere_med", "sphere_top"):
            r = 0.026  # 5.2 cm sphere
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002

        elif shape == "sphere_oversize":
            r = 0.039  # 7.8 cm jumbo sphere (> 6.2 cm max safe wrap!)
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [r, r, r]
            bbox_w, bbox_d, bbox_h = 2 * r, 2 * r, 2 * r
            obj_z = bar_top_z + r + 0.002

        else:  # "horiz_capsule" (11.2 cm wide frontally, 3.6 cm top-down waist)
            r, half_len = 0.018, 0.038
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

        cradle_z = 0.1802 if ("sphere" in shape or "capsule" in shape) else 0.1700
        for g_id in self.cradle_geom_ids:
            self.model.geom_pos[g_id, 2] = cradle_z

        self.tax_name, selected_width_m, self.is_safe_to_grasp = self.select_grasp_normal_and_taxonomy(
            shape, bbox_w, bbox_d, bbox_h
        )
        self.tax = TEST_TAXONOMIES[self.tax_name]
        self.model.site_pos[self.pocket_site_id] = self.tax["pocket_pos"]

        color = (
            [0.95, 0.25, 0.20, 1.0]
            if not self.is_safe_to_grasp
            else [*self.rng.uniform([0.18, 0.30, 0.20], [0.90, 0.90, 0.95]), 1.0]
        )
        self.model.geom_rgba[self.obj_geom_id] = color

        # Compute exact Pedestal Safety Floor (z_pocket_min) so neither pinky edge (side grasp)
        # nor the extended middle fingertip (top-down grasp) ever drops below (bar_top_z + 8mm)!
        lowest_point_drop = self.compute_lowest_hand_point_offset(
            self.tax["pitch_deg"], self.tax["roll_deg"], float(self.tax["pocket_pos"][2])
        )
        self.z_pocket_min = bar_top_z + lowest_point_drop + PEDESTAL_NO_TOUCH_DELTA

        self.obj_spawn_pos = np.array([bar_x, bar_y, obj_z])
        raw_target_z = obj_z + self.tax["z_offset"]
        safe_target_z = max(raw_target_z, self.z_pocket_min)
        self.centroid_z_mismatch = safe_target_z - obj_z

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

        self.phase = "UNSAFE_SIZE_ABORT" if not self.is_safe_to_grasp else "SYNC_REACH_AND_OPEN"
        self.phase_timer = 0.0
        self.corridor_entered = False
        self.corridor_timer = 0.0
        self.corridor_start_pos = self.start_pocket_pos.copy()
        self.wrap_s = 0.0
        self.episode_start_time = self.data.time
        self.last_log_time = self.data.time

        mode_desc = "TOP-DOWN FROM UP" if self.tax["top_down_mode"] else "HORIZONTAL SIDE"
        print(
            f"\n==================== TEST #{self.test_idx}: {shape.upper()} ({self.tax_name} | {mode_desc}) ====================\n"
            f"[Perception] BBox=(W_front={bbox_w*100:.1f}cm, W_top={bbox_d*100:.1f}cm, H={bbox_h*100:.1f}cm) | "
            f"Selected Least-Width={selected_width_m*100:.1f}cm (MaxSafe={MAX_SAFE_GRASP_WIDTH_M*100:.1f}cm)\n"
            f"[Clearance]  Pedestal_Z={bar_top_z:.3f}m | Obj_Centroid_Z={obj_z:.3f}m | "
            f"Stop_Pocket_Z={safe_target_z:.3f}m (CentroidMismatch={self.centroid_z_mismatch*100:+.2f}cm, "
            f"PinkyRingLead={self.tax['pinky_ring_lead']:.2f}x)"
        )
        if not self.is_safe_to_grasp:
            print(
                f"[SAFETY WARNING] Object least grasp width ({selected_width_m*100:.1f}cm) exceeds Max Safe Wrap "
                f"({MAX_SAFE_GRASP_WIDTH_M*100:.1f}cm)! Holding standoff & skipping unsafe grasp."
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
                    if "intermediate" in body_name or "distal" in body_name:
                        for prefix in FINGER_PREFIXES:
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

        pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()
        mid_tip_pos = self.data.site_xpos[self.mid_tip_site_id].copy()
        base_R = self.data.xmat[self.hand_base_id].reshape(3, 3).copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pocket_site_id)
        J_6x7 = np.vstack([jacp, jacr])[:, self.arm_dof_adr]

        R_des, approach_dir = self.compute_grasp_orientation_with_roll(
            target_pos=self.grasp_target_pos,
            palm_pitch_deg=self.tax["pitch_deg"],
            c_arch_yaw_deg=self.tax["yaw_deg"],
            palm_roll_deg=self.tax["roll_deg"],
        )
        rot_err_norm = float(np.linalg.norm(self.ik.orientation_error(base_R, R_des)))

        relaxed_ctrl = self.tax["relaxed_ctrl"]
        open_ctrl = self.adaptive_open_ctrl
        close_ctrl = self.adaptive_close_ctrl

        # PHASE 0: Oversized Object Safety Gate
        if self.phase == "UNSAFE_SIZE_ABORT":
            self.phase_timer += dt
            standoff_pos = self.grasp_target_pos - self.tax["standoff"] * approach_dir
            standoff_pos[2] = max(standoff_pos[2], self.z_pocket_min)
            w = min(1.0, self.phase_timer / 1.5)
            sw = 0.5 * (1.0 - np.cos(np.pi * w))
            desired_pocket_pos = (1.0 - sw) * self.start_pocket_pos + sw * standoff_pos
            self.curr_hand_ctrl = (1.0 - sw) * self.episode_start_hand_ctrl + sw * open_ctrl

            if self.phase_timer >= 2.5:
                print(
                    f"[SAFE SKIP PASS] Verified oversized object ({self.shape.upper()}) rejected safely at standoff "
                    f"(pedestal_hits={self.pedestal_collisions_count}) -> Spawning Next!"
                )
                self.spawn_next_test(is_first=False)
                return

        # PHASE 1: Reach to Standoff Gate + Synchronous Widening -> Linear Corridor Entry
        elif self.phase == "SYNC_REACH_AND_OPEN":
            self.phase_timer += dt
            # For TOP_DOWN_OVERHEAD: Standoff is directly vertically ABOVE the object (+Z_world)
            # so the extended hand descends straight down without bumping the side of the object!
            if self.tax["top_down_mode"]:
                standoff_pos = self.grasp_target_pos + np.array([0.0, 0.0, self.tax["standoff"]])
            else:
                standoff_pos = self.grasp_target_pos - self.tax["standoff"] * approach_dir
                standoff_pos[2] = max(standoff_pos[2], self.z_pocket_min + 0.012)

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
                if (w >= 1.0 and standoff_err < 0.025 and rot_err_norm < 0.10) or self.phase_timer > (T_transit + 1.3):
                    self.corridor_entered = True
                    self.corridor_start_pos = pocket_pos.copy()
                    self.corridor_timer = 0.0
                    print(
                        f"[t={elapsed:4.2f}s] Standoff Gate Reached & Hand Widened -> "
                        f"{'Descending Vertically from Above (+Z)' if self.tax['top_down_mode'] else 'Gliding Horizontally from Side'}"
                    )
            else:
                self.corridor_timer += dt
                w = min(1.0, self.corridor_timer / 1.10)
                sw = 0.5 * (1.0 - np.cos(np.pi * w))
                desired_pocket_pos = (1.0 - sw) * self.corridor_start_pos + sw * self.grasp_target_pos
                desired_pocket_pos[2] = max(desired_pocket_pos[2], self.z_pocket_min)
                self.curr_hand_ctrl = open_ctrl.copy()

                dist_to_target = float(np.linalg.norm(self.grasp_target_pos - pocket_pos))
                mid_tip_gap_mm = (mid_tip_pos[2] - self.bar_top_z) * 1000.0

                # Stop immediately when either target is reached OR (in top-down mode) middle fingertip reaches tolerance!
                top_down_tip_stop = self.tax["top_down_mode"] and (mid_tip_gap_mm <= (PEDESTAL_NO_TOUCH_DELTA * 1000.0 + 3.0))
                if dist_to_target < 0.015 or top_down_tip_stop or self.corridor_timer > 1.55:
                    self.locked_pocket_pos = pocket_pos.copy()
                    self.locked_pocket_pos[2] = max(self.locked_pocket_pos[2], self.z_pocket_min)
                    self.phase = "SYNC_WRAP"
                    self.phase_timer = 0.0
                    self.wrap_s = 0.0
                    print(
                        f"[t={elapsed:4.2f}s] Tolerance Stop Reached (MidTipGap={mid_tip_gap_mm:.1f}mm, "
                        f"CentroidZ_Diff={self.centroid_z_mismatch*100:+.2f}cm) -> Closing with Pinky/Ring Base Lead!"
                    )

        # PHASE 2: Synchronous Wrap with Faster Pinky & Ring Base-Securing Lead
        elif self.phase == "SYNC_WRAP":
            self.phase_timer += dt
            scoop_z = 0.008 * max(0.0, (self.phase_timer - 0.30) / 0.70)
            desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, scoop_z])
            desired_pocket_pos[2] = max(desired_pocket_pos[2], self.z_pocket_min)

            self.wrap_s = min(1.0, self.phase_timer / 0.85)
            # Pinky (idx 0) and Ring (idx 1) move faster by pinky_ring_lead (e.g. 1.35x - 1.40x)
            # so they wrap underneath and secure the base of the sphere/cylinder before it can slip!
            s_base = min(1.0, self.wrap_s * self.tax["pinky_ring_lead"])
            s_upper = self.wrap_s

            sync_ctrl = np.zeros(6)
            # Fast base-securing digits: pinky (0) and ring (1)
            sync_ctrl[0:2] = (1.0 - s_base) * open_ctrl[0:2] + s_base * close_ctrl[0:2]
            # Upper digits & opposed thumb: middle (2), index (3), thumb_pitch (4), thumb_yaw (5)
            sync_ctrl[2:6] = (1.0 - s_upper) * open_ctrl[2:6] + s_upper * close_ctrl[2:6]

            preload = self.tax["preload"]
            for idx, prefix in enumerate(FINGER_PREFIXES):
                if tip_forces[prefix] >= 4.0 and np.isnan(self.finger_lock_ctrl[idx]):
                    self.finger_lock_ctrl[idx] = min(
                        close_ctrl[idx], sync_ctrl[idx] + preload
                    )
                if not np.isnan(self.finger_lock_ctrl[idx]):
                    sync_ctrl[idx] = self.finger_lock_ctrl[idx]

            self.curr_hand_ctrl = sync_ctrl

            if self.phase_timer > 1.10:
                self.phase = "LIFT_AND_HOLD"
                self.phase_timer = 0.0
                print(
                    f"[t={elapsed:4.2f}s] Grasp Complete -> Lifting & Holding for {self.hold_duration:.1f}s! "
                    f"Contacts={set(contacts)} | PedestalHits={self.pedestal_collisions_count}"
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
            xy_drift = float(np.linalg.norm(obj_pos[:2] - self.obj_spawn_pos[:2])) * 100.0
            mid_tip_gap_mm = (mid_tip_pos[2] - self.bar_top_z) * 1000.0
            print(
                f"[DIAG t={elapsed:4.1f}s | {self.phase:<19}] "
                f"pos_err={pos_err*100:4.2f}cm | rot_err={rot_err:4.3f}rad | "
                f"obj_z={obj_pos[2]:.3f}m | tip_gap={mid_tip_gap_mm:4.1f}mm | "
                f"pinky={self.curr_hand_ctrl[0]:.2f} | idx={self.curr_hand_ctrl[3]:.2f} | thumb_p={self.curr_hand_ctrl[4]:.2f} | "
                f"contacts={len(contacts)} | ped_hits={self.pedestal_collisions_count}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["suite", "single"], default="suite")
    parser.add_argument(
        "--shape",
        choices=[
            "sphere_med", "sphere_small", "sphere_top", "sphere_oversize",
            "horiz_capsule", "cylinder", "box", "slim_rod"
        ],
        default="sphere_med",
    )
    parser.add_argument("--angle", type=float, default=20.0)
    parser.add_argument("--hold", type=float, default=5.0)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    fr3_scene_xml = root_dir / "fr3_sim" / "mujoco_menagerie" / "franka_fr3" / "scene.xml"
    inspire_left_xml = root_dir / "fr3_sim" / "h1_mujoco" / "inspire" / "inspire_left.xml"

    screen_w, screen_h = get_dynamic_screen_size()
    model = build_clean_test_scene(fr3_scene_xml, inspire_left_xml)
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    tester = AdaptivePerceptionGraspTester(
        model,
        data,
        mode=args.mode,
        shape=args.shape,
        angle_deg=args.angle,
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