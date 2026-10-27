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
    """Dynamically query the current monitor's resolution via GLFW."""
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
    """Dynamically resize and center the MuJoCo window via wmctrl."""
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


def build_augmented_model(scene_xml_path: Path) -> mujoco.MjModel:
    """
    Reverted to the clean Video 3 setup:
    1. Green handover box on collision bitmask 2.
    2. Simple Orange EE Sphere + Touch Sensor on fr3_link7 (collision bitmask 2).
    3. RGB-D workspace camera.
    """
    spec = mujoco.MjSpec.from_file(str(scene_xml_path))

    # 1. Mocap handover target object
    obj_body = spec.worldbody.add_body(
        name="handover_object",
        mocap=True,
        pos=[0.62, 0.0, 0.45]
    )
    obj_body.add_geom(
        name="handover_object_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.035, 0.035, 0.07],
        rgba=[0.1, 0.85, 0.3, 1.0],
        group=0,
        contype=2,
        conaffinity=2
    )

    # 2. RGB-D camera overlooking the workspace
    spec.worldbody.add_camera(
        name="rgbd_camera",
        pos=[1.20, 0.0, 0.58],
        xyaxes=[0.0, 1.0, 0.0, -0.35, 0.0, 0.936],
        fovy=58.0
    )

    # 3. Simple Orange EE Sphere & Touch Sensor on fr3_link7
    ee_body = None
    for candidate in ["fr3_link7", "fr3_link8", "link7"]:
        try:
            ee_body = spec.body(candidate)
        except Exception:
            ee_body = getattr(spec, "find_body", lambda x: None)(candidate)
        if ee_body is not None:
            print(f"[MjSpec] Attached Orange EE Sphere & Touch Sensor to body: '{candidate}'")
            break

    if ee_body is not None:
        flange_offset = [0.0, 0.0, 0.14] if ee_body.name == "fr3_link7" else [0.0, 0.0, 0.035]

        ee_body.add_site(
            name="ee_touch_site",
            pos=flange_offset,
            size=[0.042, 0.042, 0.042],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[1.0, 0.35, 0.0, 0.3],
            group=0
        )
        ee_body.add_geom(
            name="ee_palm_bumper",
            pos=flange_offset,
            size=[0.040, 0.040, 0.040],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=[1.0, 0.35, 0.0, 0.70],
            group=0,
            contype=2,
            conaffinity=2,
            mass=0.01
        )
        spec.add_sensor(
            name="ee_touch_sensor",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="ee_touch_site"
        )

    return spec.compile()


class FR3HandoverMuJoCoNode(Node):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, screen_res: tuple[int, int]):
        super().__init__('fr3_mujoco_node')
        self.model = model
        self.data = data

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.centroid_pub = self.create_publisher(PointStamped, '/handover/object_centroid', 10)
        self.depth_pub = self.create_publisher(Image, '/rgbd/depth_raw', 5)

        # FR3 7-DoF joints
        self.joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        self.qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        # Joint limits (allow joint6 full range up to 3.0 rad so the wrist can pitch horizontal)
        raw_min = np.array([self.model.jnt_range[self.model.joint(n).id, 0] for n in self.joint_names])
        raw_max = np.array([self.model.jnt_range[self.model.joint(n).id, 1] for n in self.joint_names])
        self.q_min = raw_min + 0.04
        self.q_max = raw_max - 0.04

        self.max_joint_vel = np.array([1.2, 1.2, 1.2, 1.4, 1.5, 1.5, 1.5])

        # IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_touch_site')
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_link7')
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'handover_object')
        self.obj_mocap_id = self.model.body_mocapid[self.obj_body_id]
        self.touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'ee_touch_sensor')

        # Comfortable elbow-up posture where joint4 + joint6 = -1.50 + 3.07 = +1.57 rad (90 deg horizontal!)
        self.home_q = np.array([0.0, -0.35, 0.0, -1.55, 0.0, 2.95, 0.785398])
        for adr, val in zip(self.qpos_adr, self.home_q):
            self.data.qpos[adr] = val
        self.q_des = self.home_q.copy()
        self.data.ctrl[:7] = self.q_des
        mujoco.mj_forward(self.model, self.data)

        # Handover & Safety Parameters
        self.standoff_distance = 0.13  # 13 cm standoff between sphere center and box centroid
        self.force_threshold = 1.0
        self.contact_latched = False

        # RGB-D Renderer
        self.enable_rgbd = True
        self.cam_w, self.cam_h = 128, 96
        self.rgbd_renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
        self.rgbd_renderer.enable_depth_rendering()

        self.get_logger().info(
            "FR3 Controller Ready | Video 3 stability + EE Axis Perpendicular to Object Centroid!"
        )

    def update_moving_object(self, sim_time: float) -> np.ndarray:
        if self.obj_mocap_id >= 0:
            x = 0.62 + 0.04 * np.sin(0.5 * sim_time)
            y = 0.16 * np.sin(0.6 * sim_time)
            z = 0.44 + 0.09 * np.cos(0.5 * sim_time)
            self.data.mocap_pos[self.obj_mocap_id] = [x, y, z]
        return self.data.xpos[self.obj_body_id].copy()

    def step_diff_ik_controller(self, object_centroid: np.ndarray, dt: float):
        # 1. Check Touch Sensor (auto-resumes as soon as contact clears!)
        touch_force = 0.0
        if self.touch_sensor_id >= 0:
            adr = self.model.sensor_adr[self.touch_sensor_id]
            touch_force = float(self.data.sensordata[adr])

        if touch_force > self.force_threshold:
            if not self.contact_latched:
                self.get_logger().warn(
                    f"External Contact Detected ({touch_force:.2f} N)! Holding position."
                )
                self.contact_latched = True
            self.q_des = np.array([self.data.qpos[i] for i in self.qpos_adr])
            self.data.ctrl[:7] = self.q_des
            return
        else:
            self.contact_latched = False

        # 2. Standoff Target Position (in front of the object along -X)
        desired_ee_pos = object_centroid + np.array([-self.standoff_distance, 0.0, 0.0])
        desired_ee_pos[0] = np.clip(desired_ee_pos[0], 0.40, 0.72)
        desired_ee_pos[1] = np.clip(desired_ee_pos[1], -0.35, 0.35)
        desired_ee_pos[2] = np.clip(desired_ee_pos[2], 0.18, 0.68)

        curr_pos = self.data.site_xpos[self.ee_site_id].copy()
        curr_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        # 3. Position Error + Pure Axis Alignment (Point EE +Z axis horizontally into the centroid)
        pos_err = desired_ee_pos - curr_pos

        # Current EE +Z axis in world coordinates (3rd column of rotation matrix)
        ee_z_axis = curr_mat[:, 2]
        # Desired pointing axis: straight horizontal forward (+X world, perpendicular to vertical object)
        desired_axis = np.array([1.0, 0.0, 0.0])
        # Shortest rotation vector that aligns ee_z_axis with desired_axis (no twist fighting!)
        rot_err = np.cross(ee_z_axis, desired_axis)

        kp_pos = 5.0
        kp_rot = 4.5
        twist_cmd = np.concatenate([kp_pos * pos_err, kp_rot * rot_err])

        max_lin_vel = 0.45
        lin_norm = np.linalg.norm(twist_cmd[:3])
        if lin_norm > max_lin_vel:
            twist_cmd[:3] *= (max_lin_vel / lin_norm)

        # 4. Compute 6x7 Spatial Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp, jacr])[:, self.dof_adr]

        # 5. Damped Least Squares + Strong Elbow-Up Nullspace Posture Control (from Video 3)
        damping = 0.08
        JJT = J @ J.T + (damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))

        q_curr = np.array([self.data.qpos[i] for i in self.qpos_adr])
        nullspace_err = self.home_q - q_curr
        # High weight on joint 3 & joint 5 keeps the elbow strictly upright and centered!
        null_weights = np.array([2.0, 2.0, 8.0, 2.5, 6.0, 2.0, 4.0])
        dq_null = (np.eye(7) - J_pinv @ J) @ (null_weights * nullspace_err)

        dq = J_pinv @ twist_cmd + dq_null
        dq = np.clip(dq, -self.max_joint_vel, self.max_joint_vel)

        self.q_des = self.q_des + dq * dt
        self.q_des = np.clip(self.q_des, q_curr - 0.18, q_curr + 0.18)
        self.q_des = np.clip(self.q_des, self.q_min, self.q_max)

        self.data.ctrl[:7] = self.q_des

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
        js.name = self.joint_names
        js.position = [float(self.data.qpos[i]) for i in self.qpos_adr]
        js.velocity = [float(self.data.qvel[i]) for i in self.dof_adr]
        js.effort = [float(self.data.qfrc_actuator[i]) for i in self.dof_adr]
        self.joint_pub.publish(js)

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = 'fr3_link0'
        odom.child_frame_id = 'fr3_link8'

        pos = self.data.site_xpos[self.ee_site_id]
        mat = self.data.site_xmat[self.ee_site_id]
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
    scene_xml = root_dir / "fr3_sim" / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

    if not scene_xml.exists():
        raise FileNotFoundError(f"Could not find scene XML at: {scene_xml}")

    screen_w, screen_h = get_dynamic_screen_size()

    model = build_augmented_model(scene_xml)
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    node = FR3HandoverMuJoCoNode(model, data, (screen_w, screen_h))

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
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.25, 0.0, 0.45]
    viewer.sync()

    wall_start = time.perf_counter()
    last_render_time = wall_start
    last_ros_time = wall_start
    last_rgbd_time = wall_start
    latest_centroid = np.array([0.62, 0.0, 0.45])

    try:
        while viewer.is_running() and rclpy.ok():
            now = time.perf_counter()
            wall_elapsed = now - wall_start

            with viewer.lock():
                steps_taken = 0
                while data.time < wall_elapsed and steps_taken < 25:
                    latest_centroid = node.update_moving_object(data.time)
                    node.step_diff_ik_controller(latest_centroid, sim_dt)
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
        node.get_logger().info("Shutting down FR3 MuJoCo simulation cleanly...")
    finally:
        node.rgbd_renderer.close()
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()