# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import vla_lab.tasks.manager_based.base_line.pick_place.mdp as mdp
from vla_lab.tasks.manager_based.base_line.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

# Cube geometry — MEASURED from the simulator (cubes settled at rest), not guessed.
# white = dex_cube_instanceable.usd @ scale 0.8 -> ~4.2 cm cube (half-height 0.021)
# green = green_block.usd @ scale 1.0          -> ~4.06 cm cube (half-height 0.0203)
# The previous _WHITE_HALF_HEIGHT = 0.055 was wrong by +0.034 m, which placed the goal
# 3.4 cm ABOVE the real resting height. The policy then hovered the cube in mid-air and
# every release dropped/tipped it -> catastrophic reward swing -> PPO collapse.
_GREEN_HALF_HEIGHT = 0.0203
_WHITE_HALF_HEIGHT = 0.021
# White cube center z when stacked on green cube: 0.0203 + 0.0203 + 0.021 = 0.0616
_PLACE_TARGET_Z = _GREEN_HALF_HEIGHT + _GREEN_HALF_HEIGHT + _WHITE_HALF_HEIGHT  # 0.0616
# Lift gate: above table-rest (0.021) but below stacked height (0.0616) so the gate is
# satisfied both while carrying and once correctly placed.
_LIFT_GATE_HEIGHT = 0.04


@configclass
class FrankaPickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # Inherits full Franka Lift setup: robot, white cube, ee_frame, actions, command body_name
        super().__post_init__()

        # Longer episode: pick-and-place requires more steps than pure lifting
        self.episode_length_s = 8.0

        # ── Scene: add kinematic green target cube ────────────────────────────
        # Placed at positive y so white cube spawn range (y ≤ 0.05) never overlaps
        self.scene.green_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GreenCube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.15, _GREEN_HALF_HEIGHT], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                ),
                semantic_tags=[("class", "green_cube")],
            ),
        )

        # ── Command: fix target to the stacking position ──────────────────────
        # Robot frame ≈ world frame (robot at origin). Target = white cube center
        # when resting on green cube.
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)
        self.commands.object_pose.ranges.pos_y = (0.15, 0.15)
        self.commands.object_pose.ranges.pos_z = (_PLACE_TARGET_Z, _PLACE_TARGET_Z)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        # ── Events: prevent white cube from spawning on top of green cube ─────
        # White cube init_state is [0.5, 0, 0.055]. Offset y kept ≤ 0.05 so
        # world y ≤ 0.05, giving ≥ 0.10 m gap from green cube at y = 0.15.
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.05), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        # ── Rewards: gate goal-tracking on a real lift to prevent table-sliding ──
        # White rests on the table at z = 0.021 and stacks at z = 0.0616, so the gate
        # must sit between them. 0.04 forces the cube off the table before goal-tracking
        # pays, yet still pays once correctly placed (0.0616 > 0.04). The previous 0.07
        # was ABOVE the real stacked height -> a correct placement scored nothing.
        self.rewards.object_goal_tracking.params["minimal_height"] = _LIFT_GATE_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = _LIFT_GATE_HEIGHT

        # Continuous release guidance: gradient toward opening gripper near target.
        # std=0.08 limits effective range to ~15 cm so carry-phase grasping is unaffected.
        # A velocity gate inside the reward means it only pays once the cube is actually
        # settling on the target, which suppresses the open/close jitter.
        self.rewards.gripper_open_at_goal = RewTerm(
            func=mdp.gripper_open_at_goal,
            params={
                "std": 0.08,
                "minimal_height": _LIFT_GATE_HEIGHT,
                "vel_std": 0.1,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
            weight=10.0,
        )

        # NOTE: the earlier negative "holding_closed_at_goal" penalty was removed. With the
        # corrected target height a release now actually succeeds, so positive shaping
        # (gripper_open_at_goal + place_bonus) plus the removed success-termination already
        # make "release" the higher-value behaviour. Negative shaping at the release
        # subgoal mainly added variance and accelerated the collapse, so it is not needed.

        # Binary place bonus: cube at stacking position AND gripper open. threshold=0.045
        # gives the policy a real "landing zone" instead of a knife-edge target: 0.03 was so
        # tight that a slightly off-center (but genuinely stacked) cube fell outside it, so
        # the policy kept micro-nudging the cube to chase the bonus -> the jitter you saw.
        self.rewards.place_bonus = RewTerm(
            func=mdp.object_placed_at_goal,
            params={
                "threshold": 0.045,
                "upright_error_threshold": 0.35,
                "open_threshold": 0.035,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
            weight=20.0,
        )

        # ── Observations: add green cube position ────────────────────────────
        # Gives the policy a direct signal for where to place the cube,
        # complementing the command (target_object_position) already in the obs.
        self.observations.policy.green_cube_position = ObsTerm(
            func=mdp.green_cube_position_in_robot_root_frame
        )

        # ── Terminations: end the episode once the cube is placed and settled ──
        # IMPORTANT: registered with time_out=True on purpose. A *hard* terminal
        # (time_out=False) zeroes the value bootstrap, so "finishing" would destroy the
        # remaining per-step reward stream (~37/step held closed -> ~1850 with gamma=0.98)
        # and the agent would relearn to never release (the original bug). Flagging it as a
        # truncation makes RSL-RL bootstrap the value at the success state, so terminating
        # is NOT seen as a loss. The agent still prefers to place because the placed-open
        # state simply pays more per step (place_bonus + gripper_open_at_goal) than holding
        # closed, and on success the episode resets. This gives clean place->reset episodes
        # without the don't-release pathology.
        #
        # Tolerances loosened to match a real placement: threshold 0.03 -> 0.045 (same landing
        # zone as place_bonus) and vel_threshold 0.05 -> 0.15. The old 5 cm/s was stricter than
        # the residual wobble a hovering arm imparts on the cube, so "settled" never latched and
        # the episode never ended. 0.15 m/s still requires the cube to be essentially at rest.
        self.terminations.task_success = DoneTerm(
            func=mdp.object_placed_success,
            time_out=True,
            params={
                "threshold": 0.045,
                "vel_threshold": 0.15,
                "upright_error_threshold": 0.35,
                "open_threshold": 0.035,
                "green_half_height": _GREEN_HALF_HEIGHT,
                "white_half_height": _WHITE_HALF_HEIGHT,
            },
        )


@configclass
class FrankaPickPlaceEnvCfg_PLAY(FrankaPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
