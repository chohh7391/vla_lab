# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from vla_lab.tasks.direct.base_line.automate.assembly_tasks_cfg import ASSET_DIR, AssemblyTask, Hole8mm, Peg8mm

@configclass
class Insertion(AssemblyTask):
    name = "insertion"

    assembly_id = '00015'
    assembly_dir = f"{ASSET_DIR}/{assembly_id}/"

    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    plug_grasp_json = f"{ASSET_DIR}/plug_grasps.json"
    disassembly_dist_json = f"{ASSET_DIR}/disassembly_dist.json"
    disassembly_path_json = f"{assembly_dir}/disassemble_traj.json"

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    hand_width_max: float = 0.080  # maximum opening width of gripper

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0
    fixed_asset_z_offset: float = 0.1435

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    engage_height_thresh: float = 0.01
    success_height_thresh: float = 0.003
    close_error_thresh: float = 0.015

    fixed_asset: ArticulationCfg = ArticulationCfg(
        # fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{fixed_asset_cfg.usd_path}",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    # held_asset: ArticulationCfg = ArticulationCfg(
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{held_asset_cfg.usd_path}",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        # init_state=ArticulationCfg.InitialStateCfg(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos={},
            # joint_vel={}
        ),
        # actuators={}
    )

    action_penalty_asset_scale: float = 0.001
    contact_penalty_scale: float = 0.01
