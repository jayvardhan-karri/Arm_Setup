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
    spec = mujoco.MjSpec.from_file(str(fr3_scene_xml))
    hand_spec = mujoco.MjSpec.from_file(str(inspire_xml))

    # Upgrade MuJoCo physics solver to elliptic friction cone + high impratio + noslip solver
    # This eliminates vertical micro-slipping and object tilting after grasp!
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.noslip_iterations = 3

    # Remove worldbody lights from hand_spec that use targetbodycom
    for light in list(hand_spec.worldbody.lights):
        hand_spec.delete(light)

    # Make visual geoms visible in group 0 and enable high-friction 6D collision on hand geoms
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
                # [sliding, torsional, rolling] friction
                geom.friction = [2.8, 0.08, 0.04]

    # Grasp Pocket Site inside the C-arch of the Inspire Left Hand ('base' frame):
    # Default pos is set for POWER_WRAP; dynamically adjusted per grasp taxonomy in controller!
    hand_base = hand_spec.body("base")
    hand_base.add_site(
        name="grasp_pocket_site",
        pos=[0.035, -0.015, 0.105],
        size=[0.006, 0.006, 0.006],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=[1.0, 0.2, 0.2, 0.65],
        group=0
    )

    # Touch sensor site & high-friction rubberized palm contact pad
    hand_base.add_site(
        name="palm_touch_site",
        pos=[0.020, -0.005, 0.102],
        size=[0.010, 0.028, 0.038],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=[0.0, 0.0, 0.0, 0.0],
        group=4
    )
    hand_base.add_geom(
        name="palm_contact_pad",
        pos=[0.018, -0.005, 0.102],
        size=[0.006, 0.026, 0.036],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=[1.0, 0.4, 0.0, 0.0],
        group=4,
        contype=2,
        conaffinity=2,
        condim=6,
        solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.005, 1.0],
        friction=[3.0, 0.10, 0.05],
        mass=0.005
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

    # 1. Support Pedestal / Bar with High-Friction Coaster + Subtle Centering Cup for Spheres
    pedestal_body = spec.worldbody.add_body(
        name="support_bar",
        mocap=True,
        pos=[0.62, 0.0, 0.18]
    )
    pedestal_body.add_geom(
        name="support_bar_top",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.046, 0.006, 0.01],
        pos=[0.0, 0.0, 0.174],
        rgba=[0.35, 0.38, 0.42, 1.0],
        group=0,
        contype=2,
        conaffinity=2,
        condim=6,
        # High sliding, torsional, and rolling friction so spheres & cylinders sit rock-solid
        friction=[1.8, 0.06, 0.06]
    )
    # 4 subtle 1.5mm rubber nubs forming a shallow cradle in the center of the coaster so spheres don't roll off
    for idx, ang in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
        pedestal_body.add_geom(
            name=f"support_bar_lip_{idx}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.002, 0.006, 0.002],
            pos=[0.016 * np.cos(ang), 0.016 * np.sin(ang), 0.1805],
            quat=[np.cos((ang + np.pi / 2) / 2), 0.0, 0.0, np.sin((ang + np.pi / 2) / 2)],
            rgba=[0.25, 0.28, 0.32, 1.0],
            group=0,
            contype=2,
            conaffinity=2,
            condim=6,
            friction=[1.8, 0.06, 0.06]
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

    # 2. Dynamic Target Object with 6-DoF FreeJoint
    obj_body = spec.worldbody.add_body(
        name="handover_object",
        pos=[0.62, 0.0, 0.44]
    )
    obj_body.add_freejoint(name="handover_object_joint")
    obj_body.add_geom(
        name="handover_object_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.018, 0.085, 0.018],
        rgba=[0.1, 0.85, 0.3, 1.0],
        mass=0.065,
        group=0,
        contype=3,
        conaffinity=3,
        condim=6,
        solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
        solref=[0.005, 1.0],
        friction=[2.8, 0.08, 0.05]
    )

    # 3. RGB-D camera overlooking the expanded workspace
    spec.worldbody.add_camera(
        name="rgbd_camera",
        pos=[1.30, 0.0, 0.62],
        xyaxes=[0.0, 1.0, 0.0, -0.35, 0.0, 0.936],
        fovy=60.0
    )

    return spec.compile()


# Adaptive Grasp Taxonomy Presets
# Actuator order: [pinky, ring, middle, index, thumb_proximal, thumb_yaw]
GRASP_TAXONOMIES = {
    "POWER_WRAP": {
        "pocket_local_pos": np.array([0.034, -0.015, 0.103]),  # Deep against inner palm pad
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.95]),
        "cage_ctrl": np.array([0.50, 0.50, 0.50, 0.50, 0.25, 1.22]),
        "close_ctrl": np.array([1.65, 1.65, 1.65, 1.65, 0.78, 1.25]),
        "contact_stop_preload": 0.28,  # Extra squeeze radians after finger contacts object
    },
    "PRECISION_PINCH": {
        "pocket_local_pos": np.array([0.044, -0.022, 0.136]),  # Distal thumb-index-middle pinch point
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.05]),
        "cage_ctrl": np.array([0.10, 0.10, 0.55, 0.55, 0.35, 1.28]),
        "close_ctrl": np.array([0.20, 0.20, 1.55, 1.55, 0.82, 1.30]),
        "contact_stop_preload": 0.22,
    },
    "SPHERICAL_CAGE": {
        "pocket_local_pos": np.array([0.038, -0.012, 0.112]),  # Centered inside 5-finger dome
        "open_ctrl": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.90]),
        "cage_ctrl": np.array([0.45, 0.45, 0.45, 0.45, 0.30, 1.15]),
        "close_ctrl": np.array([1.50, 1.50, 1.50, 1.50, 0.75, 1.18]),
        "contact_stop_preload": 0.25,
    },
}


class FR3InspireHandoverNode(Node):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        super().__init__('fr3_inspire_handover_node')
        self.model = model
        self.data = data
        self.rng = np.random.default_rng(42)

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
        raw_max = np.array([self.model.jnt_range[self.model.joint(n).id, 1] for n in self.arm_joint_names]
        )
        self.q_min = raw_min + 0.03
        self.q_max = raw_max - 0.03
        self.max_joint_vel = np.array([1.5, 1.5, 1.5, 1.7, 1.9, 1.9, 1.9])

        # 2. Inspire 6-DoF Hand Actuators: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        self.hand_act_names = [
            'rh_pinky', 'rh_ring', 'rh_middle', 'rh_index', 'rh_thumb_proximal', 'rh_thumb_yaw'
        ]
        self.hand_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.hand_act_names
        ]
        # Finger prefix mapping for per-finger adaptive force/contact stopping
        self.finger_prefixes = ['rh_pinky', 'rh_ring', 'rh_middle', 'rh_index', 'rh_thumb']

        # 3. Model IDs
        self.pocket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'rh_grasp_pocket_site')
        self.hand_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rh_base')
        self.bar_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'support_bar')
        self.bar_mocap_id = self.model.body_mocapid[self.bar_body_id]

        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'handover_object')
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'handover_object_geom')
        self.obj_jnt_qpos_adr = self.model.joint('handover_object_joint').qposadr[0]
        self.obj_jnt_dof_adr = self.model.joint('handover_object_joint').dofadr[0]

        self.touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'rh_palm_touch_sensor')

        # Retracted Starting Posture & Dynamic Mid-Range Nullspace Posture
        self.home_q = np.array([0.0, -0.65, 0.0, -2.25, 0.0, 1.60, -0.785])
        self.null_q = np.array([0.0, -0.25, 0.0, -1.65, 0.0, 1.55, -0.785])

        # State machine & adaptive grasp variables
        self.grasp_type = "POWER_WRAP"
        self.curr_hand_ctrl = GRASP_TAXONOMIES["POWER_WRAP"]["open_ctrl"].copy()
        self.finger_contact_lock_ctrl = np.full(6, np.nan)

        self.phase = "HUMAN_C_REACH"
        self.phase_timer = 0.0
        self.episode_start_time = 0.0
        self.last_diag_time = 0.0
        self.first_contact_logged = False
        self.spawn_idx = 0
        self.target_centroid = np.array([0.62, 0.0, 0.42])
        self.locked_pocket_pos = np.array([0.62, 0.0, 0.42])

        # Spawn first object on the bar
        self.spawn_random_object_on_bar(0.0)

        # RGB-D Renderer
        self.enable_rgbd = True
        self.cam_w, self.cam_h = 128, 96
        self.rgbd_renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
        self.rgbd_renderer.enable_depth_rendering()

    def get_object_hand_contacts_and_forces(self) -> tuple[list[str], dict[str, float]]:
        """
        Returns:
          1. List of hand body names in contact with handover_object_geom
          2. Dict mapping each finger prefix ('rh_pinky', ..., 'rh_thumb', 'rh_base') to its normal contact force (N)
        """
        hand_contacts = []
        finger_forces = {p: 0.0 for p in self.finger_prefixes + ['rh_base']}
        efc_force = np.zeros(6)

        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            if con.geom1 == self.obj_geom_id or con.geom2 == self.obj_geom_id:
                other_geom = con.geom2 if con.geom1 == self.obj_geom_id else con.geom1
                body_id = self.model.geom_bodyid[other_geom]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or body_name
                if "support_bar" not in geom_name and "floor" not in geom_name and "world" not in body_name:
                    label = body_name if body_name else geom_name
                    hand_contacts.append(label)
                    mujoco.mj_contactForce(self.model, self.data, c_idx, efc_force)
                    normal_f = float(abs(efc_force[0]))
                    for prefix in finger_forces:
                        if label.startswith(prefix):
                            finger_forces[prefix] += normal_f
        return hand_contacts, finger_forces

    def spawn_random_object_on_bar(self, sim_time: float):
        """
        Spawns objects at an increased radial distance (0.58m - 0.67m) across Front, Left, and Right sectors,
        cycling through Cylinders, Boxes, Slim Rods (Pinch), and Spheres (Spherical Cage).
        """
        # Cycle through Front, Left, and Right sectors + randomized offset for full workspace coverage
        sector_angles = [0.0, 0.55, -0.55, 0.30, -0.35, 0.70, -0.65]
        base_angle = sector_angles[self.spawn_idx % len(sector_angles)]
        angle_xy = float(np.clip(base_angle + self.rng.uniform(-0.10, 0.10), -0.78, 0.78))
        # Increased radial distance: 0.58m to 0.67m from FR3 base
        radius_xy = float(self.rng.uniform(0.58, 0.67))

        bar_x = radius_xy * np.cos(angle_xy)
        bar_y = radius_xy * np.sin(angle_xy)
        bar_top_z = float(self.rng.uniform(0.30, 0.40))
        self.data.mocap_pos[self.bar_mocap_id] = [bar_x, bar_y, bar_top_z - 0.18]

        # Cycle through the 4 object classes so all adaptive grasps (Power, Pinch, Spherical) are tested
        obj_classes = ["cylinder", "box", "sphere", "slim_rod"]
        shape_choice = obj_classes[self.spawn_idx % len(obj_classes)]
        self.spawn_idx += 1

        if shape_choice == "cylinder":
            half_width = float(self.rng.uniform(0.017, 0.020))   # 3.4 - 4.0 cm wide
            half_height = float(self.rng.uniform(0.075, 0.090))  # 15 - 18 cm tall
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [half_width, half_height, half_width]
            self.grasp_type = "POWER_WRAP"
            obj_z = bar_top_z + half_height + 0.001

        elif shape_choice == "box":
            half_width = float(self.rng.uniform(0.016, 0.019))   # 3.2 - 3.8 cm wide
            half_height = float(self.rng.uniform(0.075, 0.090))
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[self.obj_geom_id] = [half_width, half_width, half_height]
            self.grasp_type = "POWER_WRAP"
            obj_z = bar_top_z + half_height + 0.001

        elif shape_choice == "sphere":
            radius = float(self.rng.uniform(0.024, 0.028))       # 4.8 - 5.6 cm ball
            half_width, half_height = radius, radius
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
            self.model.geom_size[self.obj_geom_id] = [radius, radius, radius]
            self.grasp_type = "SPHERICAL_CAGE"
            obj_z = bar_top_z + radius + 0.001

        else:  # "slim_rod" (pen / marker / thin tool -> triggers PRECISION_PINCH)
            half_width = float(self.rng.uniform(0.009, 0.011))   # 1.8 - 2.2 cm slim profile
            half_height = float(self.rng.uniform(0.080, 0.095))
            self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[self.obj_geom_id] = [half_width, half_height, half_width]
            self.grasp_type = "PRECISION_PINCH"
            obj_z = bar_top_z + half_height + 0.001

        # Update grasp_pocket_site local coordinates to match selected Grasp Taxonomy
        preset = GRASP_TAXONOMIES[self.grasp_type]
        self.model.site_pos[self.pocket_site_id] = preset["pocket_local_pos"]

        color = self.rng.uniform([0.15, 0.25, 0.15], [0.95, 0.95, 0.95])
        self.model.geom_rgba[self.obj_geom_id] = [color[0], color[1], color[2], 1.0]

        # Rotate home_q[0] and null_q[0] to face the spawn sector (Left / Front / Right)
        start_q = self.home_q.copy()
        start_q[0] = angle_xy * 0.65
        self.null_q[0] = angle_xy

        for adr, val in zip(self.arm_qpos_adr, start_q):
            self.data.qpos[adr] = val
        for dof in self.arm_dof_adr:
            self.data.qvel[dof] = 0.0
        self.q_des = start_q.copy()
        for act_id, val in zip(self.arm_act_ids, self.q_des):
            self.data.ctrl[act_id] = val

        # Pre-shape hand according to selected grasp taxonomy
        self.curr_hand_ctrl = preset["open_ctrl"].copy()
        self.finger_contact_lock_ctrl = np.full(6, np.nan)
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val
        for j_idx in range(self.model.njnt):
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_idx) or ""
            if j_name.startswith("rh_"):
                q_adr = self.model.jnt_qposadr[j_idx]
                v_adr = self.model.jnt_dofadr[j_idx]
                self.data.qpos[q_adr] = float(preset["open_ctrl"][5]) if "yaw" in j_name else 0.0
                self.data.qvel[v_adr] = 0.0

        self.data.qpos[self.obj_jnt_qpos_adr : self.obj_jnt_qpos_adr + 3] = [bar_x, bar_y, obj_z]
        self.data.qpos[self.obj_jnt_qpos_adr + 3 : self.obj_jnt_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.obj_jnt_dof_adr : self.obj_jnt_dof_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.start_pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()
        self.target_centroid = np.array([bar_x, bar_y, obj_z])
        self.phase = "HUMAN_C_REACH"
        self.phase_timer = 0.0
        self.episode_start_time = sim_time
        self.last_diag_time = sim_time
        self.first_contact_logged = False

        sector_name = "FRONT" if abs(angle_xy) < 0.20 else ("LEFT" if angle_xy > 0 else "RIGHT")
        self.get_logger().info(
            f"\n==================== NEW EPISODE (t={sim_time:.1f}s) ====================\n"
            f"[Spawn] {shape_choice.upper()} ({sector_name}, r={radius_xy:.2f}m, ang={np.rad2deg(angle_xy):+.0f}deg) | "
            f"pos=({bar_x:.3f}, {bar_y:.3f}, {obj_z:.3f}m) | width={half_width*200:.1f}cm | "
            f"Taxonomy={self.grasp_type}"
        )

    def step_handover_controller(self, dt: float) -> np.ndarray:
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        bar_top_z = float(self.data.mocap_pos[self.bar_mocap_id, 2] + 0.18)

        elapsed = self.data.time - self.episode_start_time
        if elapsed > 11.5 or (elapsed > 1.0 and obj_pos[2] < bar_top_z - 0.12):
            if obj_pos[2] < bar_top_z - 0.12:
                self.get_logger().warn(
                    f"[Object Knocked Off] obj_z={obj_pos[2]:.3f}m < {bar_top_z - 0.12:.3f}m at t={elapsed:.1f}s -> Respawning!"
                )
            self.spawn_random_object_on_bar(self.data.time)
            obj_pos = self.data.xpos[self.obj_body_id].copy()
            elapsed = 0.0

        # Read Palm Touch Sensor & Per-Finger Contact Forces
        touch_force = 0.0
        if self.touch_sensor_id >= 0:
            adr = self.model.sensor_adr[self.touch_sensor_id]
            touch_force = float(self.data.sensordata[adr])

        hand_contacts, finger_forces = self.get_object_hand_contacts_and_forces()
        obj_drift_cm = float(np.linalg.norm(obj_pos[:2] - self.target_centroid[:2])) * 100.0

        if not self.first_contact_logged and (len(hand_contacts) > 0 or obj_drift_cm > 0.5):
            self.first_contact_logged = True
            self.get_logger().info(
                f"[FIRST CONTACT/DRIFT at t={elapsed:.2f}s in {self.phase}] "
                f"contacts={hand_contacts} | xy_drift={obj_drift_cm:.2f}cm | obj_z={obj_pos[2]:.3f}m"
            )

        curr_pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()
        curr_base_mat = self.data.xmat[self.hand_base_id].reshape(3, 3)

        # Radial horizontal vector from robot base to target object (adapts to Left, Front, and Right sectors!)
        radial_xy = np.array([self.target_centroid[0], self.target_centroid[1], 0.0])
        forward_dir = radial_xy / (np.linalg.norm(radial_xy) + 1e-6)
        up_dir = np.array([0.0, 0.0, 1.0])
        left_dir = np.cross(up_dir, forward_dir)

        # NATURAL HUMAN LEFT-HAND ORIENTATION along the object's radial ray:
        c_arch_angle = np.deg2rad(10.0)
        z_des = np.cos(c_arch_angle) * forward_dir + np.sin(c_arch_angle) * left_dir
        z_des /= np.linalg.norm(z_des)
        y_des = np.array([0.0, 0.0, -1.0])
        x_des = np.cross(y_des, z_des)
        x_des /= np.linalg.norm(x_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        rot_err = 0.5 * (
            np.cross(curr_base_mat[:, 0], R_des[:, 0]) +
            np.cross(curr_base_mat[:, 1], R_des[:, 1]) +
            np.cross(curr_base_mat[:, 2], R_des[:, 2])
        )
        rot_err_norm = float(np.linalg.norm(rot_err))

        preset = GRASP_TAXONOMIES[self.grasp_type]

        # 1. HUMAN_C_REACH: Smoothly lift & align to 14cm standoff along radial ray, then glide into C-arch
        if self.phase == "HUMAN_C_REACH":
            self.phase_timer += dt
            T_reach = 2.4
            u = min(1.0, self.phase_timer / T_reach)

            pre_standoff_pos = self.target_centroid - 0.14 * forward_dir
            if u < 0.50:
                w = u / 0.50
                sw = 0.5 * (1.0 - np.cos(np.pi * w))
                desired_pocket_pos = (1.0 - sw) * self.start_pocket_pos + sw * pre_standoff_pos
            else:
                w = (u - 0.50) / 0.50
                sw = 0.5 * (1.0 - np.cos(np.pi * w))
                desired_pocket_pos = (1.0 - sw) * pre_standoff_pos + sw * self.target_centroid

            target_hand_ctrl = preset["open_ctrl"].copy()
            dist_to_target = float(np.linalg.norm(self.target_centroid - curr_pocket_pos))

            if (u >= 1.0 and dist_to_target < 0.022) or dist_to_target < 0.012 or touch_force > 0.2 or self.phase_timer > 4.2:
                self.locked_pocket_pos = self.target_centroid.copy()
                self.phase = "CAGE_AND_OPPOSE"
                self.phase_timer = 0.0
                self.get_logger().info(
                    f"[Phase -> CAGE_AND_OPPOSE ({self.grasp_type})] (t={elapsed:.2f}s) "
                    f"dist={dist_to_target*100:.2f}cm | xy_drift={obj_drift_cm:.2f}cm | contacts={hand_contacts}"
                )

        # 2. CAGE_AND_OPPOSE: Gently advance 6mm along forward_dir while caging so object seats against palm!
        elif self.phase == "CAGE_AND_OPPOSE":
            desired_pocket_pos = self.locked_pocket_pos + 0.006 * forward_dir
            target_hand_ctrl = preset["cage_ctrl"].copy()
            self.phase_timer += dt
            if self.phase_timer > 0.45:
                self.phase = "GRASP_WRAP"
                self.phase_timer = 0.0
                self.get_logger().info(
                    f"[Phase -> GRASP_WRAP ({self.grasp_type})] (t={elapsed:.2f}s) "
                    f"Wrapping fingers with force-threshold control | contacts={hand_contacts}"
                )

        # 3. GRASP_WRAP: Close fingers with per-finger force-threshold stopping so inclined objects aren't pushed out
        elif self.phase == "GRASP_WRAP":
            desired_pocket_pos = self.locked_pocket_pos + 0.006 * forward_dir
            target_hand_ctrl = self.apply_force_adaptive_grasp(preset, finger_forces)
            self.phase_timer += dt
            if self.phase_timer > 1.0:
                self.phase = "LIFT_AND_HOLD"
                self.phase_timer = 0.0
                self.get_logger().info(
                    f"[Phase -> LIFT_AND_HOLD ({self.grasp_type})] (t={elapsed:.2f}s) "
                    f"Lifting object! contacts={hand_contacts} | obj_z={obj_pos[2]:.3f}m"
                )

        # 4. LIFT_AND_HOLD: Smoothly lift object 12 cm vertically off the pedestal while maintaining force-regulated grip
        elif self.phase == "LIFT_AND_HOLD":
            lift_z = min(0.12, 0.05 * self.phase_timer)
            desired_pocket_pos = self.locked_pocket_pos + 0.006 * forward_dir + np.array([0.0, 0.0, lift_z])
            target_hand_ctrl = self.apply_force_adaptive_grasp(preset, finger_forces)
            self.phase_timer += dt

        # Smoothly interpolate finger actuators
        hand_rate = 2.8 * dt
        self.curr_hand_ctrl += np.clip(
            target_hand_ctrl - self.curr_hand_ctrl, -hand_rate, hand_rate
        )
        for act_id, val in zip(self.hand_act_ids, self.curr_hand_ctrl):
            self.data.ctrl[act_id] = val

        # Task-space position & SO(3) orientation error
        pos_err = desired_pocket_pos - curr_pocket_pos

        kp_pos = 7.0
        kp_rot = 6.5
        twist_cmd = np.concatenate([kp_pos * pos_err, kp_rot * rot_err])

        max_lin_vel = 0.28
        lin_norm = np.linalg.norm(twist_cmd[:3])
        if lin_norm > max_lin_vel:
            twist_cmd[:3] *= (max_lin_vel / lin_norm)

        # 6x7 Spatial Jacobian at the Grasp Pocket Site
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pocket_site_id)
        J = np.vstack([jacp, jacr])[:, self.arm_dof_adr]

        # Damped Least Squares + Sector-Aligned Nullspace posture control
        damping = 0.05
        JJT = J @ J.T + (damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))

        q_curr = np.array([self.data.qpos[i] for i in self.arm_qpos_adr])
        nullspace_err = self.null_q - q_curr
        null_weights = np.array([1.2, 1.5, 3.0, 2.0, 2.0, 1.0, 0.5])
        dq_null = (np.eye(7) - J_pinv @ J) @ (null_weights * nullspace_err)

        dq = J_pinv @ twist_cmd + dq_null
        dq = np.clip(dq, -self.max_joint_vel, self.max_joint_vel)

        self.q_des = self.q_des + dq * dt
        self.q_des = np.clip(self.q_des, q_curr - 0.35, q_curr + 0.35)
        self.q_des = np.clip(self.q_des, self.q_min, self.q_max)

        # --- DIAGNOSTIC LOGGER (Prints every 1.0s of sim time) ---
        if self.data.time - self.last_diag_time >= 1.0:
            self.last_diag_time = self.data.time
            pos_err_cm = float(np.linalg.norm(pos_err)) * 100.0
            pocket_to_obj_cm = float(np.linalg.norm(obj_pos - curr_pocket_pos)) * 100.0

            hit_min = [self.arm_joint_names[i] for i in range(7) if abs(q_curr[i] - self.q_min[i]) < 0.02]
            hit_max = [self.arm_joint_names[i] for i in range(7) if abs(q_curr[i] - self.q_max[i]) < 0.02]
            jnt_lim_str = f"min:{hit_min} max:{hit_max}" if (hit_min or hit_max) else "OK"

            self.get_logger().info(
                f"[DIAG t={elapsed:4.1f}s | {self.phase:<15}] "
                f"pos_err={pos_err_cm:5.2f}cm | rot_err={rot_err_norm:5.3f}rad | "
                f"pocket->obj={pocket_to_obj_cm:5.2f}cm | drift={obj_drift_cm:4.2f}cm | obj_z={obj_pos[2]:.3f}m | "
                f"touch={touch_force:4.2f}N | contacts={len(hand_contacts)} | jnts={jnt_lim_str}"
            )

        for act_id, val in zip(self.arm_act_ids, self.q_des):
            self.data.ctrl[act_id] = val

        return obj_pos

    def apply_force_adaptive_grasp(self, preset: dict, finger_forces: dict[str, float]) -> np.ndarray:
        """
        Implements per-finger contact-force threshold stopping:
        When an individual finger exceeds 3.5 N of contact force against the object, its target actuator
        angle locks at (current_ctrl + contact_stop_preload) so it maintains firm preload without over-curling
        or pushing inclined objects into an unstable pinch!
        """
        cmd = preset["close_ctrl"].copy()
        preload = preset["contact_stop_preload"]
        force_thresh = 3.5  # Newtons per finger

        for idx, prefix in enumerate(self.finger_prefixes):
            f_val = finger_forces.get(prefix, 0.0)
            if f_val > force_thresh and np.isnan(self.finger_contact_lock_ctrl[idx]):
                self.finger_contact_lock_ctrl[idx] = min(
                    cmd[idx], self.curr_hand_ctrl[idx] + preload
                )
            if not np.isnan(self.finger_contact_lock_ctrl[idx]):
                cmd[idx] = self.finger_contact_lock_ctrl[idx]
        return cmd

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
        odom.child_frame_id = 'rh_grasp_pocket_site'

        pos = self.data.site_xpos[self.pocket_site_id]
        mat = self.data.site_xmat[self.pocket_site_id]
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
    viewer.cam.distance = 1.75
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