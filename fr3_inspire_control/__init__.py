# #!/usr/bin/env python3
# import subprocess
# from pathlib import Path
# import numpy as np
# import glfw
# import mujoco

# from .config import (
#     ARM_JOINT_NAMES,
#     HAND_ACTUATOR_NAMES,
#     FINGER_PREFIXES,
#     GRASP_TAXONOMIES,
#     DEFAULT_HOME_Q,
# )


# def get_dynamic_screen_size(default_w: int = 1920, default_h: int = 1080) -> tuple[int, int]:
#     try:
#         if glfw.init():
#             monitor = glfw.get_primary_monitor()
#             if monitor:
#                 mode = glfw.get_video_mode(monitor)
#                 if mode and mode.size.width > 0 and mode.size.height > 0:
#                     return int(mode.size.width), int(mode.size.height)
#     except Exception:
#         pass
#     return default_w, default_h


# def fit_window_to_screen(screen_w: int, screen_h: int, scale: float = 0.85):
#     win_w = int(screen_w * scale)
#     win_h = int(screen_h * scale)
#     pos_x = int((screen_w - win_w) / 2)
#     pos_y = int((screen_h - win_h) / 2)

#     try:
#         subprocess.run(
#             ["wmctrl", "-r", "MuJoCo", "-b", "remove,maximized_vert,maximized_horz,fullscreen"],
#             check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
#         subprocess.run(
#             ["wmctrl", "-r", "MuJoCo", "-e", f"0,{pos_x},{pos_y},{win_w},{win_h}"],
#             check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#         )
#     except FileNotFoundError:
#         pass


# def build_fr3_with_inspire_hand(fr3_scene_xml: Path, inspire_xml: Path) -> mujoco.MjModel:
#     spec = mujoco.MjSpec.from_file(str(fr3_scene_xml))
#     hand_spec = mujoco.MjSpec.from_file(str(inspire_xml))

#     # Proven elliptic cone + high impratio for zero-slip precision pinch & power wrap
#     spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
#     spec.option.impratio = 10.0
#     spec.option.noslip_iterations = 2

#     for light in list(hand_spec.worldbody.lights):
#         hand_spec.delete(light)

#     for body in hand_spec.bodies:
#         for geom in body.geoms:
#             if geom.classname.name == "visual" or (geom.contype == 0 and geom.conaffinity == 0):
#                 geom.group = 0
#                 geom.rgba = [0.18, 0.18, 0.20, 1.0]
#             else:
#                 geom.contype = 2
#                 geom.conaffinity = 2
#                 geom.condim = 6
#                 geom.solimp = [0.97, 0.995, 0.0005, 0.5, 2.0]
#                 geom.solref = [0.005, 1.0]
#                 geom.friction = [2.4, 0.06, 0.03]

#     hand_base = hand_spec.body("base")
#     hand_base.add_site(
#         name="grasp_pocket_site",
#         pos=[0.038, -0.015, 0.106],
#         size=[0.006, 0.006, 0.006],
#         type=mujoco.mjtGeom.mjGEOM_SPHERE,
#         rgba=[1.0, 0.2, 0.2, 0.65],
#         group=0
#     )

#     hand_base.add_site(
#         name="palm_touch_site",
#         pos=[0.015, -0.005, 0.102],
#         size=[0.008, 0.026, 0.036],
#         type=mujoco.mjtGeom.mjGEOM_BOX,
#         rgba=[0.0, 0.0, 0.0, 0.0],
#         group=4
#     )
#     # Recessed flush with the black palm mesh (x=0.012, thickness=3mm) so objects NEVER look like they float in air!
#     hand_base.add_geom(
#         name="palm_contact_pad",
#         pos=[0.012, -0.005, 0.102],
#         size=[0.003, 0.024, 0.034],
#         type=mujoco.mjtGeom.mjGEOM_BOX,
#         rgba=[0.18, 0.18, 0.20, 1.0],
#         group=0,
#         contype=2,
#         conaffinity=2,
#         condim=6,
#         solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
#         solref=[0.005, 1.0],
#         friction=[2.5, 0.08, 0.04],
#         mass=0.005
#     )
#     hand_spec.add_sensor(
#         name="palm_touch_sensor",
#         type=mujoco.mjtSensor.mjSENS_TOUCH,
#         objtype=mujoco.mjtObj.mjOBJ_SITE,
#         objname="palm_touch_site"
#     )

#     ee_body = None
#     for candidate in ["fr3_link7", "fr3_link8", "link7"]:
#         try:
#             ee_body = spec.body(candidate)
#         except Exception:
#             ee_body = getattr(spec, "find_body", lambda x: None)(candidate)
#         if ee_body is not None:
#             print(f"[MjSpec] Mounting Inspire Hand onto '{candidate}'")
#             break

#     if ee_body is None:
#         raise RuntimeError("Could not find fr3_link7 in FR3 XML!")

#     mount_site = ee_body.add_site(
#         name="inspire_mount_site",
#         pos=[0.0, 0.0, 0.107],
#         quat=[1.0, 0.0, 0.0, 0.0]
#     )
#     spec.attach(hand_spec, site=mount_site, prefix="rh_", suffix="")

#     # 1. Slim Support Pedestal (moderate 0.65 tabletop friction -> no sticky base tilting!)
#     pedestal_body = spec.worldbody.add_body(
#         name="support_bar",
#         mocap=True,
#         pos=[0.62, 0.0, 0.18]
#     )
#     pedestal_body.add_geom(
#         name="support_bar_top",
#         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
#         size=[0.027, 0.006, 0.01],
#         pos=[0.0, 0.0, 0.174],
#         rgba=[0.35, 0.38, 0.42, 1.0],
#         group=0,
#         contype=1,
#         conaffinity=1,
#         condim=6,
#         friction=[0.65, 0.02, 0.01]
#     )
#     for idx, ang in enumerate([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]):
#         pedestal_body.add_geom(
#             name=f"support_bar_lip_{idx}",
#             type=mujoco.mjtGeom.mjGEOM_CAPSULE,
#             size=[0.0018, 0.005, 0.0018],
#             pos=[0.014 * np.cos(ang), 0.014 * np.sin(ang), 0.1804],
#             quat=[np.cos((ang + np.pi / 2) / 2), 0.0, 0.0, np.sin((ang + np.pi / 2) / 2)],
#             rgba=[0.25, 0.28, 0.32, 1.0],
#             group=0,
#             contype=1,
#             conaffinity=1,
#             condim=6,
#             friction=[0.65, 0.02, 0.01]
#         )
#     pedestal_body.add_geom(
#         name="support_bar_pillar",
#         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
#         size=[0.012, 0.17, 0.01],
#         pos=[0.0, 0.0, 0.0],
#         rgba=[0.25, 0.27, 0.30, 1.0],
#         group=0,
#         contype=0,
#         conaffinity=0
#     )

#     # 2. Dynamic Target Object
#     obj_body = spec.worldbody.add_body(
#         name="handover_object",
#         pos=[0.62, 0.0, 0.44]
#     )
#     obj_body.add_freejoint(name="handover_object_joint")
#     obj_body.add_geom(
#         name="handover_object_geom",
#         type=mujoco.mjtGeom.mjGEOM_CYLINDER,
#         size=[0.018, 0.085, 0.018],
#         rgba=[0.1, 0.85, 0.3, 1.0],
#         mass=0.065,
#         group=0,
#         contype=3,
#         conaffinity=3,
#         condim=6,
#         solimp=[0.97, 0.995, 0.0005, 0.5, 2.0],
#         solref=[0.005, 1.0],
#         friction=[2.2, 0.06, 0.03]
#     )

#     # 3. RGB-D camera
#     spec.worldbody.add_camera(
#         name="rgbd_camera",
#         pos=[1.30, 0.0, 0.62],
#         xyaxes=[0.0, 1.0, 0.0, -0.35, 0.0, 0.936],
#         fovy=60.0
#     )

#     return spec.compile()


# class MuJoCoSimBackend:
#     """
#     Hardware-agnostic simulation interface for the FR3 Arm + Inspire Hand + Pedestal Spawner.
#     """
#     def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, seed: int = 42):
#         self.model = model
#         self.data = data
#         self.rng = np.random.default_rng(seed)

#         self.arm_qpos_adr = [self.model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
#         self.arm_dof_adr = [self.model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
#         self.arm_act_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
#             for n in ARM_JOINT_NAMES
#         ]

#         raw_min = np.array([self.model.jnt_range[self.model.joint(n).id, 0] for n in ARM_JOINT_NAMES])
#         raw_max = np.array([self.model.jnt_range[self.model.joint(n).id, 1] for n in ARM_JOINT_NAMES])
#         self.q_min = raw_min + 0.03
#         self.q_max = raw_max - 0.03

#         self.hand_act_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
#             for n in HAND_ACTUATOR_NAMES
#         ]

#         self.pocket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rh_grasp_pocket_site")
#         self.hand_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rh_base")
#         self.bar_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "support_bar")
#         self.bar_mocap_id = self.model.body_mocapid[self.bar_body_id]

#         self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "handover_object")
#         self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "handover_object_geom")
#         self.obj_jnt_qpos_adr = self.model.joint("handover_object_joint").qposadr[0]
#         self.obj_jnt_dof_adr = self.model.joint("handover_object_joint").dofadr[0]

#         self.touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "rh_palm_touch_sensor")
#         self.spawn_idx = 0

#     def get_arm_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         q = np.array([self.data.qpos[i] for i in self.arm_qpos_adr])
#         dq = np.array([self.data.qvel[i] for i in self.arm_dof_adr])
#         tau = np.array([self.data.qfrc_actuator[i] for i in self.arm_dof_adr])
#         return q, dq, tau

#     def get_pocket_pose_and_jacobian(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         pocket_pos = self.data.site_xpos[self.pocket_site_id].copy()
#         base_R = self.data.xmat[self.hand_base_id].reshape(3, 3).copy()

#         jacp = np.zeros((3, self.model.nv))
#         jacr = np.zeros((3, self.model.nv))
#         mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pocket_site_id)
#         J_6x7 = np.vstack([jacp, jacr])[:, self.arm_dof_adr]
#         return pocket_pos, base_R, J_6x7

#     def get_palm_touch_force(self) -> float:
#         if self.touch_sensor_id >= 0:
#             adr = self.model.sensor_adr[self.touch_sensor_id]
#             return float(self.data.sensordata[adr])
#         return 0.0

#     def get_object_contacts_and_forces(self) -> tuple[list[str], dict[str, float]]:
#         hand_contacts = []
#         finger_forces = {p: 0.0 for p in FINGER_PREFIXES + ["rh_base"]}
#         efc_force = np.zeros(6)

#         for c_idx in range(self.data.ncon):
#             con = self.data.contact[c_idx]
#             if con.geom1 == self.obj_geom_id or con.geom2 == self.obj_geom_id:
#                 other_geom = con.geom2 if con.geom1 == self.obj_geom_id else con.geom1
#                 body_id = self.model.geom_bodyid[other_geom]
#                 body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
#                 geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or body_name
#                 if "support_bar" not in geom_name and "floor" not in geom_name and "world" not in body_name:
#                     label = body_name if body_name else geom_name
#                     hand_contacts.append(label)
#                     mujoco.mj_contactForce(self.model, self.data, c_idx, efc_force)
#                     normal_f = float(abs(efc_force[0]))
#                     for prefix in finger_forces:
#                         if label.startswith(prefix):
#                             finger_forces[prefix] += normal_f
#         return hand_contacts, finger_forces

#     def send_commands(self, q_arm_des: np.ndarray, ctrl_hand_des: np.ndarray):
#         for act_id, val in zip(self.arm_act_ids, q_arm_des):
#             self.data.ctrl[act_id] = val
#         for act_id, val in zip(self.hand_act_ids, ctrl_hand_des):
#             self.data.ctrl[act_id] = val

#     def spawn_next_episode(self) -> dict:
#         is_first_episode = (self.spawn_idx == 0)
#         sector_angles = [0.0, 0.55, -0.55, 0.30, -0.35, 0.70, -0.65]
#         base_angle = sector_angles[self.spawn_idx % len(sector_angles)]
#         angle_xy = float(np.clip(base_angle + self.rng.uniform(-0.10, 0.10), -0.78, 0.78))
#         radius_xy = float(self.rng.uniform(0.58, 0.67))

#         bar_x = radius_xy * np.cos(angle_xy)
#         bar_y = radius_xy * np.sin(angle_xy)
#         bar_top_z = float(self.rng.uniform(0.30, 0.40))
#         self.data.mocap_pos[self.bar_mocap_id] = [bar_x, bar_y, bar_top_z - 0.18]

#         obj_classes = ["cylinder", "box", "sphere", "slim_rod"]
#         shape_choice = obj_classes[self.spawn_idx % len(obj_classes)]
#         self.spawn_idx += 1

#         if shape_choice == "cylinder":
#             half_width = float(self.rng.uniform(0.017, 0.020))
#             half_height = float(self.rng.uniform(0.075, 0.090))
#             self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
#             self.model.geom_size[self.obj_geom_id] = [half_width, half_height, half_width]
#             grasp_type = "POWER_WRAP"
#             obj_z = bar_top_z + half_height + 0.001
#         elif shape_choice == "box":
#             half_width = float(self.rng.uniform(0.016, 0.019))
#             half_height = float(self.rng.uniform(0.075, 0.090))
#             self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_BOX
#             self.model.geom_size[self.obj_geom_id] = [half_width, half_width, half_height]
#             grasp_type = "POWER_WRAP"
#             obj_z = bar_top_z + half_height + 0.001
#         elif shape_choice == "sphere":
#             radius = float(self.rng.uniform(0.024, 0.028))
#             half_width, half_height = radius, radius
#             self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_SPHERE
#             self.model.geom_size[self.obj_geom_id] = [radius, radius, radius]
#             grasp_type = "SPHERICAL_CAGE"
#             obj_z = bar_top_z + radius + 0.001
#         else:
#             half_width = float(self.rng.uniform(0.009, 0.011))
#             half_height = float(self.rng.uniform(0.080, 0.095))
#             self.model.geom_type[self.obj_geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
#             self.model.geom_size[self.obj_geom_id] = [half_width, half_height, half_width]
#             grasp_type = "PRECISION_PINCH"
#             obj_z = bar_top_z + half_height + 0.001

#         tax = GRASP_TAXONOMIES[grasp_type]
#         self.model.site_pos[self.pocket_site_id] = tax.pocket_local_pos

#         color = self.rng.uniform([0.15, 0.25, 0.15], [0.95, 0.95, 0.95])
#         self.model.geom_rgba[self.obj_geom_id] = [color[0], color[1], color[2], 1.0]

#         obj_width_m = 2.0 * half_width
#         open_ctrl = tax.open_ctrl.copy()

#         # Only teleport to home on Episode 0; on Episode >= 2, plan smoothly from current arm pose!
#         if is_first_episode:
#             start_q = DEFAULT_HOME_Q.copy()
#             start_q[0] = angle_xy * 0.65
#             for adr, val in zip(self.arm_qpos_adr, start_q):
#                 self.data.qpos[adr] = val
#             for dof in self.arm_dof_adr:
#                 self.data.qvel[dof] = 0.0
#             for j_idx in range(self.model.njnt):
#                 j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_idx) or ""
#                 if j_name.startswith("rh_"):
#                     q_adr = self.model.jnt_qposadr[j_idx]
#                     v_adr = self.model.jnt_dofadr[j_idx]
#                     self.data.qpos[q_adr] = float(open_ctrl[5]) if "yaw" in j_name else float(open_ctrl[0])
#                     self.data.qvel[v_adr] = 0.0
#             self.send_commands(start_q, open_ctrl)
#         else:
#             start_q, _, _ = self.get_arm_state()

#         # Respawn object cleanly on the pedestal
#         self.data.qpos[self.obj_jnt_qpos_adr : self.obj_jnt_qpos_adr + 3] = [bar_x, bar_y, obj_z]
#         self.data.qpos[self.obj_jnt_qpos_adr + 3 : self.obj_jnt_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
#         self.data.qvel[self.obj_jnt_dof_adr : self.obj_jnt_dof_adr + 6] = 0.0

#         mujoco.mj_forward(self.model, self.data)

#         # RealSense 2.5D Perceived Feature Summary:
#         # Front surface depth + perceived width/height above pedestal plane -> symmetric grasp center
#         radial_unit = np.array([np.cos(angle_xy), np.sin(angle_xy), 0.0])
#         front_surface_pos = np.array([bar_x, bar_y, obj_z]) - half_width * radial_unit
#         symmetric_grasp_center = front_surface_pos + (0.5 * obj_width_m) * radial_unit

#         sector_name = "FRONT" if abs(angle_xy) < 0.20 else ("LEFT" if angle_xy > 0 else "RIGHT")
#         return {
#             "shape": shape_choice,
#             "sector_name": sector_name,
#             "angle_xy": angle_xy,
#             "radius_xy": radius_xy,
#             "bar_top_z": bar_top_z,
#             "obj_spawn_pos": symmetric_grasp_center,
#             "width_m": obj_width_m,
#             "width_cm": obj_width_m * 100.0,
#             "height_cm": half_height * 200.0,
#             "grasp_type": grasp_type,
#             "open_ctrl": open_ctrl,
#             "start_q": start_q,
#             "start_pocket_pos": self.data.site_xpos[self.pocket_site_id].copy(),
#         }