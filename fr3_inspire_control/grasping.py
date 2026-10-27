#!/usr/bin/env python3
import numpy as np
from .config import (
    ARM_JOINT_NAMES,
    FINGER_PREFIXES,
    GRASP_TAXONOMIES,
    GraspTaxonomyConfig,
)
from .kinematics import DiffIKController
from .mujoco_sim import MuJoCoSimBackend


class HandoverGraspCoordinator:
    """
    Synchronous Human-Like Grasp Coordinator (Verified across Cylinder, Box, Slim Rod, Sphere):
      1. SYNC_REACH_AND_OPEN: Cylindrical arc transit to standoff gate while synchronously
         widening the 4-finger set and thumb from relaxed_ctrl -> open_ctrl, followed by
         linear corridor entry.
      2. SYNC_WRAP: Single normalized closure parameter s in [0, 1] drives all 4 fingers
         and the thumb simultaneously at matched normalized velocity, locking individual digits
         when their distal/intermediate pads sense >= 4.0N grip force.
      3. LIFT_AND_HOLD: Lifts object 12 cm vertically and holds for 5.0 seconds before
         smoothly transitioning from current arm pose to the next spawned object.
    """
    def __init__(self, sim: MuJoCoSimBackend, ik: DiffIKController, logger, hold_duration: float = 5.0):
        self.sim = sim
        self.ik = ik
        self.logger = logger
        self.hold_duration = hold_duration

        self.phase = "SYNC_REACH_AND_OPEN"
        self.phase_timer = 0.0
        self.corridor_entered = False
        self.corridor_start_pos = np.zeros(3)
        self.corridor_timer = 0.0
        self.wrap_s = 0.0

        self.episode_start_time = 0.0
        self.last_diag_time = 0.0

        self.ep_info: dict = {}
        self.taxonomy: GraspTaxonomyConfig = GRASP_TAXONOMIES["POWER_WRAP"]
        self.q_des = np.zeros(7)
        self.curr_hand_ctrl = np.zeros(6)
        self.episode_start_hand_ctrl = np.zeros(6)
        self.finger_lock_ctrl = np.full(6, np.nan)

        self.grasp_target_pos = np.zeros(3)
        self.locked_pocket_pos = np.zeros(3)

        self.reset_episode(0.0)

    def reset_episode(self, sim_time: float):
        is_first = (self.sim.spawn_idx == 0)
        self.ep_info = self.sim.spawn_next_episode()
        self.taxonomy = GRASP_TAXONOMIES[self.ep_info["grasp_type"]]
        relaxed_ctrl = self.ep_info["relaxed_ctrl"]

        if is_first:
            self.q_des = self.ep_info["start_q"].copy()
            self.curr_hand_ctrl = relaxed_ctrl.copy()
        else:
            self.q_des, _, _ = self.sim.get_arm_state()

        self.episode_start_hand_ctrl = self.curr_hand_ctrl.copy()
        self.finger_lock_ctrl = np.full(6, np.nan)

        spawn_pos = self.ep_info["obj_spawn_pos"].copy()
        self.grasp_target_pos = spawn_pos + np.array([0.0, 0.0, self.taxonomy.target_z_offset])
        self.locked_pocket_pos = self.grasp_target_pos.copy()

        self.phase = "SYNC_REACH_AND_OPEN"
        self.phase_timer = 0.0
        self.corridor_entered = False
        self.corridor_timer = 0.0
        self.corridor_start_pos = self.ep_info["start_pocket_pos"].copy()
        self.wrap_s = 0.0
        self.episode_start_time = sim_time
        self.last_diag_time = sim_time

        pos = self.ep_info["obj_spawn_pos"]
        self.logger.info(
            f"\n==================== EPISODE #{self.sim.spawn_idx}: {self.ep_info['shape'].upper()} ({self.taxonomy.name}) ====================\n"
            f"[Spawn] ({self.ep_info['sector_name']}, r={self.ep_info['radius_xy']:.2f}m, ang={np.rad2deg(self.ep_info['angle_xy']):+.1f}deg) | "
            f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}m) | W={self.ep_info['width_cm']:.1f}cm | HoldAfterGrasp={self.hold_duration:.1f}s"
        )

    def step(self, dt: float) -> np.ndarray:
        sim_time = self.sim.data.time
        obj_pos = self.sim.data.xpos[self.sim.obj_body_id].copy()
        bar_top_z = self.ep_info["bar_top_z"]
        elapsed = sim_time - self.episode_start_time

        # Early respawn if object is ever knocked below the pedestal coaster
        if elapsed > 1.0 and obj_pos[2] < bar_top_z - 0.10:
            self.logger.warn(
                f"[Object Dropped at t={elapsed:.2f}s] obj_z={obj_pos[2]:.3f}m -> Advancing to next episode!"
            )
            self.reset_episode(sim_time)
            return self.sim.data.xpos[self.sim.obj_body_id].copy()

        hand_contacts, tip_forces = self.sim.get_object_contacts_and_forces()
        spawn_pos = self.ep_info["obj_spawn_pos"]
        obj_drift_cm = float(np.linalg.norm(obj_pos[:2] - spawn_pos[:2])) * 100.0

        curr_pocket_pos, curr_base_R, J_6x7 = self.sim.get_pocket_pose_and_jacobian()

        R_des, approach_dir = self.ik.compute_grasp_orientation(
            target_pos=self.grasp_target_pos,
            palm_pitch_deg=self.taxonomy.palm_pitch_deg,
            c_arch_yaw_deg=self.taxonomy.c_arch_yaw_deg,
        )
        rot_err_vec = self.ik.orientation_error(curr_base_R, R_des)
        rot_err_norm = float(np.linalg.norm(rot_err_vec))

        relaxed_ctrl = self.taxonomy.relaxed_ctrl
        open_ctrl = self.taxonomy.open_ctrl
        close_ctrl = self.taxonomy.close_ctrl

        # PHASE 1: Cylindrical Arc Reach + Synchronous Widening -> Linear Corridor Glide
        if self.phase == "SYNC_REACH_AND_OPEN":
            self.phase_timer += dt
            standoff_pos = self.grasp_target_pos - self.taxonomy.standoff_dist * approach_dir
            start_pos = self.ep_info["start_pocket_pos"]

            if not self.corridor_entered:
                transit_dist = float(np.linalg.norm(standoff_pos - start_pos))
                T_transit = float(np.clip(transit_dist / 0.35, 1.30, 2.20))
                w = min(1.0, self.phase_timer / T_transit)
                sw = 0.5 * (1.0 - np.cos(np.pi * w))

                if w < 0.35:
                    s_rel = 0.5 * (1.0 - np.cos(np.pi * (w / 0.35)))
                    self.curr_hand_ctrl = (1.0 - s_rel) * self.episode_start_hand_ctrl + s_rel * relaxed_ctrl
                else:
                    s_open = 0.5 * (1.0 - np.cos(np.pi * ((w - 0.35) / 0.65)))
                    self.curr_hand_ctrl = (1.0 - s_open) * relaxed_ctrl + s_open * open_ctrl

                r_start = float(np.hypot(start_pos[0], start_pos[1]))
                r_end = float(np.hypot(standoff_pos[0], standoff_pos[1]))
                th_start = float(np.arctan2(start_pos[1], start_pos[0]))
                th_end = float(np.arctan2(standoff_pos[1], standoff_pos[0]))

                r_curr = (1.0 - sw) * r_start + sw * r_end
                th_curr = (1.0 - sw) * th_start + sw * th_end
                z_curr = (1.0 - sw) * start_pos[2] + sw * standoff_pos[2] + 0.04 * np.sin(np.pi * sw)
                desired_pocket_pos = np.array([r_curr * np.cos(th_curr), r_curr * np.sin(th_curr), z_curr])

                standoff_err = float(np.linalg.norm(standoff_pos - curr_pocket_pos))
                if (w >= 1.0 and standoff_err < 0.022 and rot_err_norm < 0.08) or self.phase_timer > (T_transit + 1.2):
                    self.corridor_entered = True
                    self.corridor_start_pos = curr_pocket_pos.copy()
                    self.corridor_timer = 0.0
                    self.logger.info(
                        f"[t={elapsed:4.2f}s] Standoff Gate Reached & Hand Synchronously Widened -> Entering Corridor"
                    )
            else:
                self.corridor_timer += dt
                w = min(1.0, self.corridor_timer / 1.0)
                sw = 0.5 * (1.0 - np.cos(np.pi * w))
                desired_pocket_pos = (1.0 - sw) * self.corridor_start_pos + sw * self.grasp_target_pos
                self.curr_hand_ctrl = open_ctrl.copy()

                dist_to_target = float(np.linalg.norm(self.grasp_target_pos - curr_pocket_pos))
                if dist_to_target < 0.015 or self.corridor_timer > 1.5:
                    self.locked_pocket_pos = self.grasp_target_pos.copy()
                    self.phase = "SYNC_WRAP"
                    self.phase_timer = 0.0
                    self.wrap_s = 0.0
                    self.logger.info(
                        f"[t={elapsed:4.2f}s] Object Inside Pocket (err={dist_to_target*100:.2f}cm) "
                        f"-> Starting Simultaneous Finger+Thumb Closure!"
                    )

        # PHASE 2: Simultaneous Normalized Finger + Thumb Closure (s in [0, 1])
        elif self.phase == "SYNC_WRAP":
            self.phase_timer += dt
            unweight_z = min(0.005, 0.005 * (self.phase_timer / 0.6))
            desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, unweight_z])

            self.wrap_s = min(1.0, self.phase_timer / 0.85)
            sync_ctrl = (1.0 - self.wrap_s) * open_ctrl + self.wrap_s * close_ctrl

            preload = self.taxonomy.contact_stop_preload
            for idx, prefix in enumerate(FINGER_PREFIXES):
                if tip_forces[prefix] >= 4.0 and np.isnan(self.finger_lock_ctrl[idx]):
                    self.finger_lock_ctrl[idx] = min(
                        close_ctrl[idx], sync_ctrl[idx] + preload
                    )
                if not np.isnan(self.finger_lock_ctrl[idx]):
                    sync_ctrl[idx] = self.finger_lock_ctrl[idx]

            self.curr_hand_ctrl = sync_ctrl

            if self.phase_timer > 1.05:
                self.phase = "LIFT_AND_HOLD"
                self.phase_timer = 0.0
                self.logger.info(
                    f"[t={elapsed:4.2f}s] Grasp Complete -> Lifting & Holding for {self.hold_duration:.1f}s! "
                    f"Contacts={set(hand_contacts)} | "
                    f"TipForces={ {k: round(v, 1) for k, v in tip_forces.items() if k != 'rh_base'} }"
                )

        # PHASE 3: Lift 12 cm & Hold for 5.0 Seconds Before Spawning Next Object
        else:
            self.phase_timer += dt
            lift_z = 0.005 + min(0.12, 0.060 * self.phase_timer)
            desired_pocket_pos = self.locked_pocket_pos + np.array([0.0, 0.0, lift_z])

            if self.phase_timer >= self.hold_duration:
                lifted_cm = (obj_pos[2] - spawn_pos[2]) * 100.0
                status = "PASS" if (lifted_cm > 6.0 and len(hand_contacts) >= 2) else "DROP"
                self.logger.info(
                    f"[{status}] Held {self.ep_info['shape'].upper()} for {self.hold_duration:.1f}s "
                    f"(lifted={lifted_cm:+.1f}cm, contacts={len(hand_contacts)}) -> Spawning Next!"
                )
                self.reset_episode(sim_time)
                return self.sim.data.xpos[self.sim.obj_body_id].copy()

        # Step Limit-Aware Diff-IK
        q_curr, _, _ = self.sim.get_arm_state()
        self.q_des, pos_err_m, rot_err_rad = self.ik.step_ik(
            q_curr=q_curr,
            q_des_prev=self.q_des,
            curr_pos=curr_pocket_pos,
            curr_R=curr_base_R,
            des_pos=desired_pocket_pos,
            des_R=R_des,
            J_6x7=J_6x7,
            sector_angle=self.ep_info["angle_xy"],
            max_lin_vel=0.42,
            dt=dt,
        )

        self.sim.send_commands(self.q_des, self.curr_hand_ctrl)

        # 1.0 Hz Diagnostic Telemetry
        if sim_time - self.last_diag_time >= 1.0:
            self.last_diag_time = sim_time
            pos_err_cm = pos_err_m * 100.0

            hit_min = [ARM_JOINT_NAMES[i] for i in range(7) if abs(q_curr[i] - self.sim.q_min[i]) < 0.02]
            hit_max = [ARM_JOINT_NAMES[i] for i in range(7) if abs(q_curr[i] - self.sim.q_max[i]) < 0.02]
            jnt_lim_str = f"min:{hit_min} max:{hit_max}" if (hit_min or hit_max) else "OK"

            self.logger.info(
                f"[DIAG t={elapsed:4.1f}s | {self.phase:<19}] "
                f"pos_err={pos_err_cm:4.2f}cm | rot_err={rot_err_rad:4.3f}rad | "
                f"obj_z={obj_pos[2]:.3f}m | drift={obj_drift_cm:4.2f}cm | "
                f"thumb_p={self.curr_hand_ctrl[4]:.2f} | idx={self.curr_hand_ctrl[3]:.2f} | "
                f"contacts={len(hand_contacts)} | jnts={jnt_lim_str}"
            )

        return obj_pos