#!/usr/bin/env python3
import os
import subprocess

# Fix Ubuntu 22.04 Wayland/XWayland DPI & OpenGL startup glitches
os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"

import time
from pathlib import Path
import numpy as np
import glfw
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


def get_dynamic_screen_size(default_w: int = 1920, default_h: int = 1080) -> tuple[int, int]:
    try:
        if glfw.init():
            monitor = glfw.get_primary_monitor()
            if monitor:
                mode = glfw.get_video_mode(monitor)
                if mode and mode.size.width > 0 and mode.size.height > 0:
                    return int(mode.size.width), int(mode.size.height)
    except Exception:
        pass
    return default_w, default_h


def fit_window_to_screen(screen_w: int, screen_h: int, scale: float = 0.85):
    win_w = int(screen_w * scale)
    win_h = int(screen_h * scale)
    pos_x = int((screen_w - win_w) / 2)
    pos_y = int((screen_h - win_h) / 2)

    try:
        subprocess.run(
            ["wmctrl", "-r", "MuJoCo", "-b", "remove,maximized_vert,maximized_horz,fullscreen"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["wmctrl", "-r", "MuJoCo", "-e", f"0,{pos_x},{pos_y},{win_w},{win_h}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass


def build_fr3_with_inspire_hand(fr3_scene_xml: Path, inspire_xml: Path) -> mujoco.MjModel:
    """
    Merges Franka FR3 scene.xml with Inspire Left Hand XML:
    1. Invisible Palm Touch Sensor (no orange box blocking the palm).
    2. Movable Pedestal Bar (mocap body) + Dynamic Graspable Object (6-DoF freejoint) resting on top.
    3. High-friction finger & object contact settings for secure grasping & lifting.
    """
    spec = mujoco.MjSpec.from_file(str(fr3_scene_xml))
    hand_spec = mujoco.MjSpec.from_file(str(inspire_xml))

    # Remove worldbody lights from hand_spec that use targetbodycom
    for light in list(hand_spec.worldbody.lights):
        hand_spec.delete(light)

    # Make visual geoms visible in group 0 and enable collision bitmask 2 on collision geoms
    for body in hand_spec.bodies:
        for geom in body.geoms:
            if geom.classname.name == "visual" or (geom.contype == 0 and geom.conaffinity == 0):
                geom.group = 0
                geom.rgba = [0.18, 0.18, 0.20, 1.0]
            else:
                geom.contype = 2
                geom.conaffinity = 2
                geom.friction = [1.5, 0.02, 0.005]

    # Add an INVISIBLE Palm Touch Sensor Site wrapping the built-in palm collision boxes
    hand_base = hand_spec.body("base")
    hand_base.add_site(
        name="palm_touch_site",
        pos=[0.018, -0.003, 0.095],  # Palm surface center
        size=[0.014, 0.028, 0.038],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=[0.0, 0.0, 0.0, 0.0],   # Completely invisible! No orange box!
        group=4                      # Hidden group
    )
    hand_spec.add_sensor(
        name="palm_touch_sensor",
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="palm_touch_site"
    )

    # Mount Inspire Hand onto fr3_link7
    ee_body = None
    for candidate in ["fr3_link7", "fr3_link8", "link7"]:
        try:
            ee_body = spec.body(candidate)
        except Exception:
            ee_body = getattr(spec, "find_body", lambda x: None)(candidate)
        if ee_body is not None:
            print(f"[MjSpec] Mounting Inspire Hand onto '{candidate}'")
            break

    if ee_body is None:
        raise RuntimeError("Could not find fr3_link7 in FR3 XML!")

    mount_site = ee_body.add_site(
        name="inspire_mount_site",
        pos=[0.0, 0.0, 0.107],
        quat=[1.0, 0.0, 0.0, 0.0]
    )
    spec.attach(hand_spec, site=mount_site, prefix="rh_", suffix="")

    # 1. Support Pedestal / Bar (Mocap body so we can reposition its height & XY randomly)
    pedestal_body = spec.worldbody.add_body(
        name="support_bar",
        mocap=True,
        pos=[0.56, 0.0, 0.18]
    )
    # Narrow top platform + vertical stand bar so the hand fingers never hit the table
    pedestal_body.add_geom(
        name="support_bar_top",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.026, 0.006, 0.01],  # 2.6 cm radius top plate
        pos=[0.0, 0.0, 0.174],
        rgba=[0.35, 0.38, 0.42, 1.0],
        group=0,
        contype=2,
        conaffinity=2,
        friction=[0.8, 0.01, 0.002]
    )
    pedestal_body.add_geom(
        name="support_bar_pillar",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.012, 0.17, 0.01],   # Slim vertical bar supporting the top plate
        pos=[0.0, 0.0, 0.0],
        rgba=[0.25, 0.27, 0.30, 1.0],
        group=0,
        contype=0,                  # Non-colliding pillar so arm/fingers focus on the object
        conaffinity=0
    )

    # 2. Dynamic Target Object with a 6-DoF FreeJoint (sits on top of support_bar under gravity)
    obj_body = spec.worldbody.add_body(
        name="handover_object",
        pos=[0.56, 0.0, 0.44]
    )
    obj_body.add_freejoint(name="handover_object_joint")
    obj_body.add_geom(
        name="handover_object_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.021, 0.075, 0.021],
        rgba=[0.1, 0.85, 0.3, 1.0],
        mass=0.08,                  # Light 80g object for crisp grasping
        group=0,
        contype=2,
        conaffinity=2,
        condim=4,
        friction=[1.5, 0.02, 0.005]
    )

    # 3. RGB-D camera overlooking the workspace
    spec.worldbody.add_camera(
        name="rgbd_camera",
        pos=[1.20, 0.0, 0.58],
        xyaxes=[0.0, 1.0, 0.0, -0.35, 0.0, 0.936],
        fovy=58.0
    )

    return spec.compile()


class FR3InspireHandoverNode(Node):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__('fr3_inspire_handover_node')
        self.model = model
        self.data = data
        self.rng = np.random.default_rng()

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.centroid_pub = self.create_publisher(PointStamped, '/handover/object_centroid', 10)
        self.depth_pub = self.create_publisher(Image, '/rgbd/depth_raw', 5)

        # 1. FR3 7-DoF Arm Joints & Actuators
        self.arm_joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        self.arm_qpos_adr = [self.model.joint(n).qposadr[0] for n in self.arm_joint_names]
        self.arm_dof_adr = [self.model.joint(n).dofadr[0] for n in self.arm_joint_names]
        self.arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.arm_joint_names
        ]

        raw_min = np.array([self.model.jnt_range[self.model.joint(n).id, 0] for n in self.arm_joint_names])
        raw_max = np.array([self.model.jnt_range[self.model.joint(n).id, 1] for n in self.arm_joint_names])
        self.q_min = raw_min + 0.04
        self.q_max = raw_max - 0.04
        self.max_joint_vel = np.array([1.0, 1.0, 1.0, 1.2, 1.4, 1.4, 1.4])

        # 2. Inspire 6-DoF Hand Actuators
        self.hand_act_names = [
            'rh_pinky', 'rh_ring', 'rh_middle', 'rh_index', 'rh_thumb_proximal', 'rh_thumb_yaw'
        ]
        self.hand_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.hand_act_names
        ]

        # Hand Postures: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        self.hand_open_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.20])
        self.hand_close_ctrl = np.array([1.25, 1.25, 1.25, 1.25, 0.55, 1.20])
        self.curr_hand_ctrl = self.hand_open_ctrl.copy()

        # 3. Model IDs
        self.palm_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'rh_palm_touch_site')
        self.hand_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rh_base')
        self.bar_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'support_bar')
        self.bar_mocap_id = self.model.body_mocapid[self.bar_body_id]

        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'handover_object')
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'handover_object_geom')
        self.obj_jnt_qpos_adr = self.model.joint('handover_object_joint').qposadr[0]
        self.obj_jnt_dof_adr = self.model.joint('handover_object_joint').dofadr[0]

        self.touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'rh_palm_touch_sensor')

        # State machine variables
        self.phase = "STAGING"
        self.phase_timer = 0.0
        self.episode_start_time = 0.0
        self.last_diag_time = 0.0
        self.target_centroid = np.array([0.56, 0.0, 0.42])
        self.locked_pocket_pos = np.array([0.56, 0.0, 0.42])

        # Comfortable Home Posture with joint7 pre-rotated (-0.78 rad) to match Left Hand thumb-up posture!
        self.home_q = np.array([0.0, -0.40, 0.0, -1.75, 0.0, 1.55, -0.78])
        for adr, val in zip(self.arm_qpos_adr, self.home_q):
            self.data.qpos[adr] = val
        self.q_des = self.home_q.copy()

        for act_id, val in zip(self.arm_act_ids, self.q_des):
            self.data.ctrl[act_id] = val
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val

        # Spawn first random object on the bar
        self.spawn_random_object_on_bar(0.0)

        # RGB-D Renderer
        self.enable_rgbd = True
        self.cam_w, self.cam_h = 128, 96
        self.rgbd_renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
        self.rgbd_renderer.enable_depth_rendering()

    def spawn_random_object_on_bar(self, sim_time: float):
        """
        Spawns a random object (Cylinder, Box, or Capsule) with random size and color
        on a support bar at a safe distance & height inside the FR3 work envelope.
        """
        # Safe work envelope: radial distance 0.48m to 0.65m (never too close to arm base!)
        radius_xy = self.rng.uniform(0.48, 0.64)
        angle_xy = self.rng.uniform(-0.42, 0.42)  # +/- 24 degrees in front of robot
        bar_x = radius_xy * np.cos(angle_xy)
        bar_y = radius_xy * np.sin(angle_xy)

        # Random top surface height of the support bar between 0.28m and 0.46m
        bar_top_z = self.rng.uniform(0.28, 0.46)
        # In build_fr3_with_inspire_hand, support_bar_top top surface is at mocap_z + 0.18m
        self.data.mocap_pos[self.bar_mocap_id] = [bar_x, bar_y, bar_top_z - 0.18]

        # Randomize object shape, radius/width, and half-height
        shape_choice = self.rng.choice(["cylinder", "box", "capsule"])
        half_width = self.rng.uniform(0.017, 0.024)   # 3.4 cm to 4.8 cm wide (fits inside Inspire hand)
        half_height = self.rng.uniform(0.065, 0.090)  # 13 cm to 18 cm tall

        if shape_choice == "cylinder":
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [half_width, half_height, half_width]
            eff_half_h = half_height
        elif shape_choice == "box":
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [half_width, half_width, half_height]
            eff_half_h = half_height
        else:
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CAPSULE
            self.model.geom_size[self.obj_geom_id] = [half_width, half_height - half_width, half_width]
            eff_half_h = half_height

        # Random vivid color
        color = self.rng.uniform([0.1, 0.2, 0.1], [0.95, 0.95, 0.95])
        self.model.geom_rgba[self.obj_geom_id] = [color[0], color[1], color[2], 1.0]

        # Place object resting stably on top of the bar
        obj_z = bar_top_z + eff_half_h + 0.002
        self.data.qpos[self.obj_jnt_qpos_adr : self.obj_jnt_qpos_adr + 3] = [bar_x, bar_y, obj_z]
        self.data.qpos[self.obj_jnt_qpos_adr + 3 : self.obj_jnt_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.obj_jnt_dof_adr : self.obj_jnt_dof_adr + 6] = 0.0

        # Reset hand to open
        self.curr_hand_ctrl = self.hand_open_ctrl.copy()
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val

        self.obj_radius = half_width
        self.target_centroid = np.array([bar_x, bar_y, obj_z])
        self.phase = "APPROACH"
        self.phase_timer = 0.0
        self.episode_start_time = sim_time

        mujoco.mj_forward(self.model, self.data)
        self.get_logger().info(
            f"[New Spawn] {shape_choice.upper()} at pos=({bar_x:.2f}, {bar_y:.2f}, {obj_z:.2f}m) | "
            f"width={half_width*200:.1f}cm, height={eff_half_h*200:.1f}cm"
        )

    def step_handover_controller(self, dt: float) -> np.ndarray:
        # Respawn a new random object every 11 seconds or if the object gets knocked off
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        if (self.data.time - self.episode_start_time > 11.0) or (obj_pos[2] < 0.12):
            self.spawn_random_object_on_bar(self.data.time)
            obj_pos = self.data.xpos[self.obj_body_id].copy()

        # Read Palm Touch Sensor (True physical contact on the black palm)
        touch_force = 0.0
        if self.touch_sensor_id >= 0:
            adr = self.model.sensor_adr[self.touch_sensor_id]
            touch_force = float(self.data.sensordata[adr])

        curr_palm_pos = self.data.site_xpos[self.palm_site_id].copy()
        curr_base_mat = self.data.xmat[self.hand_base_id].reshape(3, 3)

        # Compute radial horizontal approach direction from robot base to object
        radial_xy = np.array([self.target_centroid[0], self.target_centroid[1], 0.0])
        approach_dir = radial_xy / (np.linalg.norm(radial_xy) + 1e-6)  # Unit vector pointing from base to object

        # Left Hand Coordinate Frame Alignment:
        # +X_base (Palm Normal) -> approach_dir (points straight into object)
        # -Y_base (Thumb side)  -> World +Z [0, 0, 1] (Thumb on top) => +Y_base = [0, 0, -1]
        # +Z_base (Fingers)     -> cross(+X_base, +Y_base)
        x_des = approach_dir
        y_des = np.array([0.0, 0.0, -1.0])
        z_des = np.cross(x_des, y_des)
        z_des /= np.linalg.norm(z_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        # State Machine: APPROACH (stop 9cm in front) -> PRE_GRASP (move into palm without knocking) -> GRASP -> LIFT
        if self.phase == "APPROACH":
            standoff = 0.095  # Stop 9.5 cm in front of object first
            desired_palm_pos = self.target_centroid - standoff * approach_dir
            target_hand_ctrl = self.hand_open_ctrl

            if np.linalg.norm(desired_palm_pos - curr_palm_pos) < 0.022:
                self.phase_timer += dt
                if self.phase_timer > 0.35:
                    self.phase = "PRE_GRASP"
                    self.phase_timer = 0.0
                    self.get_logger().info("Phase -> PRE_GRASP: Moving palm gently to object standoff...")

        elif self.phase == "PRE_GRASP":
            # Stop right where the object surface is ~3mm from the palm (or stop immediately on touch!)
            standoff = self.obj_radius + 0.006
            desired_palm_pos = obj_pos - standoff * approach_dir
            target_hand_ctrl = self.hand_open_ctrl

            dist_err = np.linalg.norm(desired_palm_pos - curr_palm_pos)
            if touch_force > 0.25 or dist_err < 0.010:
                self.phase_timer += dt
                if self.phase_timer > 0.15 or touch_force > 0.4:
                    self.phase = "GRASP"
                    self.phase_timer = 0.0
                    self.locked_palm_pos = curr_palm_pos.copy()
                    self.get_logger().info(
                        f"Phase -> GRASP: Closing fingers! (touch_force={touch_force:.2f} N)"
                    )

        elif self.phase == "GRASP":
            desired_palm_pos = self.locked_palm_pos
            target_hand_ctrl = self.hand_close_ctrl
            self.phase_timer += dt
            if self.phase_timer > 1.0:
                self.phase = "LIFT"
                self.phase_timer = 0.0
                self.get_logger().info("Phase -> LIFT: Lifting grasped object off the bar!")

        elif self.phase == "LIFT":
            # Smoothly lift 10 cm above the bar
            lift_height = min(0.10, 0.06 * self.phase_timer)
            desired_palm_pos = self.locked_palm_pos + np.array([0.0, 0.0, lift_height])
            target_hand_ctrl = self.hand_close_ctrl
            self.phase_timer += dt

        # Smoothly drive finger actuators
        hand_rate = 2.8 * dt
        self.curr_hand_ctrl += np.clip(
            target_hand_ctrl - self.curr_hand_ctrl, -hand_rate, hand_rate
        )
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val

        # Task-space position & SO(3) orientation error
        pos_err = desired_palm_pos - curr_palm_pos
        rot_err = 0.5 * (
            np.cross(curr_base_mat[:, 0], R_des[:, 0]) +
            np.cross(curr_base_mat[:, 1], R_des[:, 1]) +
            np.cross(curr_base_mat[:, 2], R_des[:, 2])
        )

        kp_pos = 4.5
        kp_rot = 4.0
        twist_cmd = np.concatenate([kp_pos * pos_err, kp_rot * rot_err])

        max_lin_vel = 0.30
        lin_norm = np.linalg.norm(twist_cmd[:3])
        if lin_norm > max_lin_vel:
            twist_cmd[:3] *= (max_lin_vel / lin_norm)

        # 6x7 Spatial Jacobian at the Palm Touch Site
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.palm_site_id)
        J = np.vstack([jacp, jacr])[:, self.arm_dof_adr]

        # Damped Least Squares + Nullspace posture control (light weight on joint 7 so wrist never fights!)
        damping = 0.08
        JJT = J @ J.T + (damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))

        q_curr = np.array([self.data.qpos[i] for i in self.arm_qpos_adr])
        nullspace_err = self.home_q - q_curr
        null_weights = np.array([1.5, 2.0, 6.0, 2.0, 4.0, 1.5, 0.5])
        dq_null = (np.eye(7) - J_pinv @ J) @ (null_weights * nullspace_err)

        dq = J_pinv @ twist_cmd + dq_null
        dq = np.clip(dq, -self.max_joint_vel, self.max_joint_vel)

        self.q_des = self.q_des + dq * dt
        self.q_des = np.clip(self.q_des, q_curr - 0.15, q_curr + 0.15)
        self.q_des = np.clip(self.q_des, self.q_min, self.q_max)

        for act_id, val in zip(self.arm_act_ids, self.q_des):
            self.data.ctrl[act_id] = val

        return obj_pos

    def publish_rgbd_depth(self, obj_centroid: np.ndarray):
        if not self.enable_rgbd:
            return
        self.rgbd_renderer.update_scene(self.data, camera="rgbd_camera")
        depth_map = self.rgbd_renderer.render()

        now_msg = self.get_clock().now().to_msg()
        depth_msg = Image()
        depth_msg.header.stamp = now_msg
        depth_msg.header.frame_id = "rgbd_camera_optical_frame"
        depth_msg.height = self.cam_h
        depth_msg.width = self.cam_w
        depth_msg.encoding = "32FC1"
        depth_msg.step = self.cam_w * 4
        depth_msg.data = depth_map.astype(np.float32).tobytes()
        self.depth_pub.publish(depth_msg)

        pt = PointStamped()
        pt.header.stamp = now_msg
        pt.header.frame_id = "fr3_link0"
        pt.point.x, pt.point.y, pt.point.z = map(float, obj_centroid)
        self.centroid_pub.publish(pt)

    def publish_telemetry(self):
        now_msg = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now_msg
        js.header.frame_id = 'fr3_link0'
        js.name = self.arm_joint_names
        js.position = [float(self.data.qpos[i]) for i in self.arm_qpos_adr]
        js.velocity = [float(self.data.qvel[i]) for i in self.arm_dof_adr]
        js.effort = [float(self.data.qfrc_actuator[i]) for i in self.arm_dof_adr]
        self.joint_pub.publish(js)

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = 'fr3_link0'
        odom.child_frame_id = 'rh_palm_touch_site'

        pos = self.data.site_xpos[self.palm_site_id]
        mat = self.data.site_xmat[self.palm_site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)

        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        self.odom_pub.publish(odom)


def main():
    rclpy.init()

    root_dir = Path(__file__).resolve().parent
    fr3_scene_xml = root_dir / "fr3_sim" / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

    # Prioritize Left Hand as requested
    inspire_left_xml = root_dir / "fr3_sim" / "h1_mujoco" / "inspire" / "inspire_left.xml"
    inspire_right_xml = root_dir / "fr3_sim" / "h1_mujoco" / "inspire" / "inspire_right.xml"

    if inspire_left_xml.exists():
        inspire_xml = inspire_left_xml
    elif inspire_right_xml.exists():
        inspire_xml = inspire_right_xml
    else:
        raise FileNotFoundError("Could not find Inspire XML in fr3_sim/h1_mujoco/inspire/")

    screen_w, screen_h = get_dynamic_screen_size()

    model = build_fr3_with_inspire_hand(fr3_scene_xml, inspire_xml)
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    node = FR3InspireHandoverNode(model, data)

    render_dt = 1.0 / 60.0
    ros_dt = 1.0 / 100.0
    rgbd_dt = 1.0 / 10.0
    sim_dt = model.opt.timestep

    viewer = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    )

    time.sleep(0.05)
    fit_window_to_screen(screen_w, screen_h, scale=0.85)

    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance = 1.7
    viewer.cam.lookat[:] = [0.35, 0.0, 0.40]
    viewer.sync()

    wall_start = time.perf_counter()
    last_render_time = wall_start
    last_ros_time = wall_start
    last_rgbd_time = wall_start
    latest_centroid = node.target_centroid.copy()

    try:
        while viewer.is_running() and rclpy.ok():
            now = time.perf_counter()
            wall_elapsed = now - wall_start

            with viewer.lock():
                steps_taken = 0
                while data.time < wall_elapsed and steps_taken < 25:
                    latest_centroid = node.step_handover_controller(sim_dt)
                    mujoco.mj_step(model, data)
                    steps_taken += 1

            now = time.perf_counter()

            if now - last_rgbd_time >= rgbd_dt:
                with viewer.lock():
                    node.publish_rgbd_depth(latest_centroid)
                last_rgbd_time = now

            if now - last_ros_time >= ros_dt:
                node.publish_telemetry()
                rclpy.spin_once(node, timeout_sec=0.0)
                last_ros_time = now

            if now - last_render_time >= render_dt:
                viewer.sync()
                last_render_time = now

            time.sleep(0.001)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutting down FR3 + Inspire Hand simulation cleanly...")
    finally:
        node.rgbd_renderer.close()
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()