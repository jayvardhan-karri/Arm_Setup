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
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry


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


def fit_window_to_screen(screen_w: int, screen_h: int, scale: float = 0.90):
    """
    Dynamically resize and center the MuJoCo window to `scale` (e.g., 85%) of the screen
    if `wmctrl` is installed, preventing the fullscreen XWayland black-screen stall.
    """
    win_w = int(screen_w * scale)
    win_h = int(screen_h * scale)
    pos_x = int((screen_w - win_w) / 2)
    pos_y = int((screen_h - win_h) / 2)

    try:
        # Remove fullscreen/maximized hints and apply centered dynamic geometry
        subprocess.run(
            ["wmctrl", "-r", "MuJoCo", "-b", "remove,maximized_vert,maximized_horz,fullscreen"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["wmctrl", "-r", "MuJoCo", "-e", f"0,{pos_x},{pos_y},{win_w},{win_h}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        # wmctrl is optional; if not installed, MuJoCo stays at default maximized size
        pass


class FR3MuJoCoNode(Node):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, screen_res: tuple[int, int]):
        super().__init__('fr3_mujoco_node')
        self.model = model
        self.data = data

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # FR3 7-DoF joints
        self.joint_names = [f'fr3_joint{i}' for i in range(1, 8)]
        self.qpos_adr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.dof_adr = [self.model.joint(name).dofadr[0] for name in self.joint_names]

        # End-effector site ('attachment_site' on fr3_link8)
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site'
        )
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_link8'
        )

        # Reset to 'home' keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            home_q = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
            for adr, val in zip(self.qpos_adr, home_q):
                self.data.qpos[adr] = val
            self.data.ctrl[:7] = home_q

        mujoco.mj_forward(self.model, self.data)

        arm_mass = float(np.sum(self.model.body_mass[1:]))
        self.get_logger().info(
            f"FR3 initialized | Mass: {arm_mass:.2f} kg | "
            f"Detected Screen: {screen_res[0]}x{screen_res[1]} | dt: {self.model.opt.timestep} s"
        )

    def publish_telemetry(self):
        now_msg = self.get_clock().now().to_msg()

        # 1. Publish JointState
        js = JointState()
        js.header.stamp = now_msg
        js.header.frame_id = 'fr3_link0'
        js.name = self.joint_names
        js.position = [float(self.data.qpos[i]) for i in self.qpos_adr]
        js.velocity = [float(self.data.qvel[i]) for i in self.dof_adr]
        js.effort = [float(self.data.qfrc_actuator[i]) for i in self.dof_adr]
        self.joint_pub.publish(js)

        # 2. Publish End-Effector Odometry
        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = 'fr3_link0'
        odom.child_frame_id = 'fr3_link8'

        if self.ee_site_id >= 0:
            pos = self.data.site_xpos[self.ee_site_id]
            mat = self.data.site_xmat[self.ee_site_id]
        else:
            pos = self.data.xpos[self.ee_body_id]
            mat = self.data.xmat[self.ee_body_id]

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)

        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])

        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_id, vel, 0
        )
        odom.twist.twist.angular.x = float(vel[0])
        odom.twist.twist.angular.y = float(vel[1])
        odom.twist.twist.angular.z = float(vel[2])
        odom.twist.twist.linear.x = float(vel[3])
        odom.twist.twist.linear.y = float(vel[4])
        odom.twist.twist.linear.z = float(vel[5])

        self.odom_pub.publish(odom)


def main():
    rclpy.init()

    root_dir = Path(__file__).resolve().parent
    scene_xml = root_dir / "fr3_sim" / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

    if not scene_xml.exists():
        raise FileNotFoundError(f"Could not find scene XML at: {scene_xml}")

    # 1. Dynamically detect current monitor resolution
    screen_w, screen_h = get_dynamic_screen_size()

    model = mujoco.MjModel.from_xml_path(str(scene_xml))

    # 2. Dynamically configure MuJoCo framebuffer to match screen resolution
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    node = FR3MuJoCoNode(model, data, (screen_w, screen_h))

    render_dt = 1.0 / 60.0   # 60 Hz viewer refresh
    ros_dt = 1.0 / 100.0     # 100 Hz ROS telemetry + spin

    viewer = mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    )

    # 3. Allow window creation, then dynamically scale window to 85% of screen & set camera
    time.sleep(0.05)
    fit_window_to_screen(screen_w, screen_h, scale=0.85)

    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.0, 0.0, 0.45]
    viewer.sync()

    last_render_time = time.perf_counter()
    last_ros_time = time.perf_counter()

    try:
        while viewer.is_running() and rclpy.ok():
            loop_start = time.perf_counter()

            with viewer.lock():
                mujoco.mj_step(model, data)

            now = time.perf_counter()

            if now - last_ros_time >= ros_dt:
                node.publish_telemetry()
                rclpy.spin_once(node, timeout_sec=0.0)
                last_ros_time = now

            if now - last_render_time >= render_dt:
                viewer.sync()
                last_render_time = now

            elapsed = time.perf_counter() - loop_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutting down FR3 MuJoCo simulation cleanly...")
    finally:
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()