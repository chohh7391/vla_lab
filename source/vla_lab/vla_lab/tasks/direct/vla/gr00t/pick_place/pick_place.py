# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.api.objects.cuboid import VisualCuboid
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.robot.manipulators.examples.franka.franka_experimental import FrankaExperimental
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.numpy.maths import matmul
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats


class FrankaPickPlace:
    """Simple, direct Franka pick-and-place controller.

    No complex inheritance, no RL concepts, no task wrappers.
    Just straightforward robot control that's easy to understand and modify.
    """

    def __init__(self, events_dt: Optional[List[float]] = None):
        """Initialize the FrankaPickPlace controller.

        Sets up initial state variables for the state machine.

        Args:
            events_dt: List of step counts for each phase. If None, uses default values.
        """
        self.cube = None
        self.robot = None

        # Define step counts for each phase
        self.events_dt = events_dt
        if self.events_dt is None:
            # Phase durations: [(x,y) positioning, approach, grasp, lift, move, release, retract]
            self.events_dt = [
                60,  # Phase 0: Move to x,y position above cube
                40,  # Phase 1: Approach down to cube
                40,  # Phase 2: Close gripper to grasp
                40,  # Phase 3: Lift cube upward
                80,  # Phase 4: Move cube to target location
                20,  # Phase 5: Open gripper to release
                20,  # Phase 6: Move up and away
            ]
        self._event = 0
        self._step = 0

    def setup_scene(
        self,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """Set up the scene with robot and cube.

        Creates and adds a Franka robot and a dynamic cube to the simulation world.
        Sets default values for positions and sizes if not provided.

        Args:
            cube_initial_position: Initial cube position [x, y, z]. Defaults to [0.5, 0.0, 0.0258].
            cube_initial_orientation: Initial cube orientation as quaternion [w, x, y, z]. Defaults to [1, 0, 0, 0].
            cube_size: Cube dimensions [w, h, d]. Defaults to [0.0515, 0.0515, 0.0515].
            target_position: Target position for placing [x, y, z]. Defaults to [-0.3, -0.3, cube_height/2].
            offset: Additional offset to apply to target position [x, y, z]. Defaults to [0, 0, 0].
        """
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            self.cube_initial_position = np.array([0.5, 0.0, 0.0258])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            self.target_position = np.array([0.6, 0.0, 0.12])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        # Create a new USD stage with default sunlight lighting
        stage_utils.create_new_stage(template="sunlight")

        # Create the Franka robot controller (inherits from Articulation)
        self.robot = FrankaExperimental(robot_path="/World/robot", create_robot=True)
        self.end_effector_link = self.robot.end_effector_link

        # Add ground plane environment for physics simulation
        ground_plane = stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # Create blue visual material for the cube
        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths="/World/Cube",
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

        # Create blue visual material for the target
        self.target_plane = VisualCuboid(
            prim_path="/World/Target",
            position=np.array([0.5, 0.0, 0.0002]),
            orientation=self.cube_initial_orientation,
            scale=np.array([0.15, 0.15, 0.0001]),
            size=1.0,
            color=np.array([1.0, 0.0, 0.0])
        )

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """Execute one step of the pick-and-place operation using the specified IK method.

        Args:
            ik_method: The inverse kinematics method to use. Defaults to "damped-least-squares".

        Returns:
            True if a step was executed, False if the sequence is complete.
        """
        if self.is_done():
            return False

        # Get downward-facing orientation for end-effector
        goal_orientation = self.robot.get_downward_orientation()

        # Phase 0: Move to x,y position above cube
        if self._event == 0:
            # if self._step == 0:
            #     print("Phase 0: Moving to x,y position above cube...")

            # Goal is above the cube at a safe height, matching x,y position
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = np.array([cube_pos[0, 0], cube_pos[0, 1], cube_pos[0, 2] + 0.2])  # Higher above cube

            # Use the new high-level method that combines position and orientation
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[0]:
                self._event += 1
                self._step = 0

        # Phase 1: Approach down to the cube
        elif self._event == 1:
            # if self._step == 0:
            #     print("Phase 1: Approaching cube...")

            # Goal is above and slightly behind the cube for proper approach
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.1])  # Approach from above with safe distance

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[1]:
                self._event += 1
                self._step = 0

        # Phase 2: Close gripper to grasp the cube
        elif self._event == 2:
            # if self._step == 0:
            #     print("Phase 2: Closing gripper...")

            # Close gripper
            self.robot.close_gripper()

            self._step += 1
            if self._step >= self.events_dt[2]:
                self._event += 1
                self._step = 0

        # Phase 3: Lift the cube
        elif self._event == 3:
            # if self._step == 0:
            #     print("Phase 3: Lifting cube...")

            # Get current end effector position and lift up
            _, current_position, _ = self.robot.get_current_state()
            goal_position = current_position + np.array([0.0, 0.0, 0.2])

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[3]:
                self._event += 1
                self._step = 0

        # Phase 4: Move cube to target location
        elif self._event == 4:
            # if self._step == 0:
            #     print("Phase 4: Moving cube...")

            # Move to position using the controller
            self.robot.set_end_effector_pose(
                position=self.target_position, orientation=goal_orientation, ik_method=ik_method
            )

            self._step += 1
            if self._step >= self.events_dt[4]:
                self._event += 1
                self._step = 0

        # Phase 5: Open gripper to release cube
        elif self._event == 5:
            # if self._step == 0:
            #     print("Phase 5: Opening gripper...")

            # Open gripper
            self.robot.open_gripper()

            self._step += 1
            if self._step >= self.events_dt[5]:
                self._event += 1
                self._step = 0

        # Phase 6: Move up
        elif self._event == 6:
            # if self._step == 0:
            #     print("Phase 6: Moving up...")

            # Goal is to lift up
            cube_pos = self.cube.get_world_poses()[0].numpy()
            goal_position = cube_pos + np.array([0.0, 0.0, 0.3])  # Move above the cube

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[6]:
                self._event += 1
                self._step = 0

        return True

    def is_done(self) -> bool:
        """Check if the pick-and-place sequence is complete.

        Returns:
            True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self.events_dt):
            return True
        else:
            return False

    def reset(self, cube_position: Optional[np.ndarray] = None, cube_orientation: Optional[np.ndarray] = None):
        """Reset the entire pick-and-place system to initial state.

        This is the main reset function that resets both robot and cube.
        Use this for complete system reset.

        Args:
            cube_position: Optional new position for the cube. If None, uses initial position.
            cube_orientation: Optional new orientation for the cube. If None, uses initial orientation.
        """
        print("Resetting pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        print("Pick-and-place system reset complete")

    def reset_robot(self):
        """Reset the robot to its default state.

        Resets the robot's joint positions to the default configuration
        and resets the state machine to the beginning.
        """
        if self.robot is not None:
            # Reset robot using the controller
            self.robot.reset_to_default_pose()

            # Reset state machine
            self._event = 0
            self._step = 0

            print("Robot reset to default state")
        else:
            print("Warning: Franka controller not initialized, cannot reset")

    def reset_cube(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        """Reset the cube to its initial position and orientation.

        Args:
            position: Optional new position for the cube. If None, uses initial position.
            orientation: Optional new orientation for the cube. If None, uses initial orientation.
        """
        if self.cube is not None:
            # Use provided position/orientation or fall back to initial values
            reset_position = position if position is not None else self.cube_initial_position
            reset_orientation = orientation if orientation is not None else self.cube_initial_orientation

            # Reset cube position and orientation
            self.cube.set_world_poses(
                positions=reset_position.reshape(1, -1), orientations=reset_orientation.reshape(1, -1)
            )

            print(f"Cube reset to position: {reset_position}")
        else:
            print("Warning: Cube not initialized, cannot reset")

    def set_target(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):

        target_position = position if position is not None else np.array([0.5, 0.0, 0.0002])
        target_orientation = orientation if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])

        self.target_plane.set_world_pose(
            position=np.array([target_position[0], target_position[1], 0.0002]), orientation=target_orientation
        )
        self.target_position = target_position
