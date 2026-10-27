#!/usr/bin/env python3
import os

# Fix Ubuntu 22.04 Wayland/XWayland DPI & OpenGL startup glitches
os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"

import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped

from fr3_inspire_control.config import ARM_JOINT_NAMES
from fr3_inspire_control.kinematics import DiffIKController
from fr3_inspire_control.mujoco_sim import (
    get_dynamic_screen_size,
    fit_window_to_screen,
    build_fr3_with_inspire_hand,
    MuJoCoSimBackend,
)
from fr3_inspire_control.grasping import HandoverGraspCoordinator


class ModularHandoverROSNode(Node):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__("fr3_inspire_handover_node")
        self.model = model
        self.data = data

        # ROS 2 Publishers
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.centroid_pub = self.create_publisher(PointStamped, "/handover/object_centroid", 10)
        self.depth_pub = self.create_publisher(Image, "/rgbd/depth_raw", 5)

        # Modular Backend, IK, and Grasp Coordinator
        self.sim = MuJoCoSimBackend(model, data, seed=42)
        self.ik = DiffIKController(self.sim.q_min, self.sim.q_max)
        self.coordinator = HandoverGraspCoordinator(self.sim, self.ik, self.get_logger())

        # Offscreen RGB-D Renderer
        self.enable_rgbd = True
        self.cam_w, self.cam_h = 128, 96
        self.rgbd_renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
        self.rgbd_renderer.enable_depth_rendering()

    def step_simulation(self, dt: float) -> np.ndarray:
        return self.coordinator.step(dt)

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
        q, dq, tau = self.sim.get_arm_state()

        js = JointState()
        js.header.stamp = now_msg
        js.header.frame_id = "fr3_link0"
        js.name = ARM_JOINT_NAMES
        js.position = [float(x) for x in q]
        js.velocity = [float(x) for x in dq]
        js.effort = [float(x) for x in tau]
        self.joint_pub.publish(js)

        odom = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id = "fr3_link0"
        odom.child_frame_id = "rh_grasp_pocket_site"

        pos = self.data.site_xpos[self.sim.pocket_site_id]
        mat = self.data.site_xmat[self.sim.pocket_site_id]
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
    fr3_scene_xml = root_dir / "fr3_sim" / "models" / "franka_fr3" / "scene.xml"
    inspire_left_xml = root_dir / "fr3_sim" / "models" / "inspire" / "inspire_left.xml"
    inspire_right_xml = root_dir / "fr3_sim" / "h1_mujoco" / "inspire" / "inspire_right.xml"

    inspire_xml = inspire_left_xml if inspire_left_xml.exists() else inspire_right_xml
    if not inspire_xml.exists():
        raise FileNotFoundError("Could not find Inspire XML in fr3_sim/h1_mujoco/inspire/")

    screen_w, screen_h = get_dynamic_screen_size()
    model = build_fr3_with_inspire_hand(fr3_scene_xml, inspire_xml)
    model.vis.global_.offwidth = screen_w
    model.vis.global_.offheight = screen_h

    data = mujoco.MjData(model)
    node = ModularHandoverROSNode(model, data)

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
    viewer.cam.distance = 1.75
    viewer.cam.lookat[:] = [0.35, 0.0, 0.40]
    viewer.sync()

    wall_start = time.perf_counter()
    last_render_time = wall_start
    last_ros_time = wall_start
    last_rgbd_time = wall_start
    latest_centroid = node.coordinator.grasp_target_pos.copy()

    try:
        while viewer.is_running() and rclpy.ok():
            now = time.perf_counter()
            wall_elapsed = now - wall_start

            with viewer.lock():
                steps_taken = 0
                while data.time < wall_elapsed and steps_taken < 25:
                    latest_centroid = node.step_simulation(sim_dt)
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
        node.get_logger().info("Shutting down modular FR3 + Inspire Hand simulation cleanly...")
    finally:
        node.rgbd_renderer.close()
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()