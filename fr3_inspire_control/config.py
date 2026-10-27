#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraspTaxonomyConfig:
    name: str
    pocket_local_pos: np.ndarray     # Site position in Inspire 'base' frame [x, y, z]
    relaxed_ctrl: np.ndarray         # Natural relaxed hand posture at start of reach
    open_ctrl: np.ndarray            # Wide-open C-preshape [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    close_ctrl: np.ndarray           # Full synchronous wrap target
    contact_stop_preload: float      # Extra squeeze radians after >= 4.0N distal pad contact
    palm_pitch_deg: float            # Downward pitch of hand (0 deg = horizontal, 20 deg = sphere dome)
    c_arch_yaw_deg: float            # Inward yaw of C-arch so thumb & fingers clear object symmetrically
    standoff_dist: float             # Approach corridor standoff distance (m)
    target_z_offset: float           # Vertical offset relative to object centroid (m)


GRASP_TAXONOMIES: dict[str, GraspTaxonomyConfig] = {
    "POWER_WRAP": GraspTaxonomyConfig(
        name="POWER_WRAP",
        pocket_local_pos=np.array([0.040, -0.020, 0.108]),
        relaxed_ctrl=np.array([0.45, 0.45, 0.45, 0.45, 0.25, 0.45]),
        open_ctrl=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.22]),
        close_ctrl=np.array([1.65, 1.65, 1.65, 1.65, 0.85, 1.26]),
        contact_stop_preload=0.25,
        palm_pitch_deg=0.0,
        c_arch_yaw_deg=8.0,
        standoff_dist=0.14,
        target_z_offset=0.015,
    ),
    "PRECISION_PINCH": GraspTaxonomyConfig(
        name="PRECISION_PINCH",
        pocket_local_pos=np.array([0.044, -0.022, 0.136]),
        relaxed_ctrl=np.array([0.35, 0.35, 0.40, 0.40, 0.20, 0.50]),
        open_ctrl=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.08]),
        close_ctrl=np.array([0.20, 0.20, 1.58, 1.58, 0.85, 1.30]),
        contact_stop_preload=0.22,
        palm_pitch_deg=0.0,
        c_arch_yaw_deg=11.0,
        standoff_dist=0.14,
        target_z_offset=0.015,
    ),
    "SPHERICAL_CAGE": GraspTaxonomyConfig(
        name="SPHERICAL_CAGE",
        pocket_local_pos=np.array([0.046, -0.022, 0.120]),
        relaxed_ctrl=np.array([0.40, 0.40, 0.40, 0.40, 0.25, 0.55]),
        open_ctrl=np.array([0.08, 0.08, 0.08, 0.08, 0.05, 1.22]),
        close_ctrl=np.array([1.48, 1.48, 1.48, 1.48, 0.85, 1.30]),
        contact_stop_preload=0.28,
        palm_pitch_deg=20.0,
        c_arch_yaw_deg=10.0,
        standoff_dist=0.14,
        target_z_offset=0.002,
    ),
}

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_ACTUATOR_NAMES = [
    "rh_pinky", "rh_ring", "rh_middle", "rh_index", "rh_thumb_proximal", "rh_thumb_yaw"
]
FINGER_PREFIXES = ["rh_pinky", "rh_ring", "rh_middle", "rh_index", "rh_thumb"]

DEFAULT_HOME_Q = np.array([0.0, -0.45, 0.0, -1.85, -0.30, 1.55, -0.785])
DEFAULT_NULL_Q = np.array([0.0, -0.15, 0.0, -1.45, -0.40, 1.45, -0.785])
MAX_ARM_JOINT_VEL = np.array([1.6, 1.6, 1.6, 1.8, 2.0, 2.0, 2.0])