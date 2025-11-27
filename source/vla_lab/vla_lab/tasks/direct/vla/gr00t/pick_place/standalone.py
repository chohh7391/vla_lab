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

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from pick_place import FrankaPickPlace
import numpy as np

def update_gr00t_observations(
    env_id, eef_pos, eef_quat, eef_pos_prev, eef_quat_prev, gripper_qpos, current_time,
):

    # calcaulte others
    delta_pos = eef_pos - eef_pos_prev
    quat_rel = eef_quat * eef_quat_prev.inv()

    delta_rot = quat_rel.as_rotvec()

    state = np.concatenate((eef_pos, eef_quat, gripper_qpos), axis=-1)
    arm_action = np.concatenate((delta_pos, delta_rot), axis=-1)




def main():
    print("Starting Simple Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = FrankaPickPlace()
    pick_place.setup_scene()

    # Play the simulation.
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()
    simulation_app.update()

    reset_needed = True

    num_envs = 4
    gr00t_data_buffers = [{} for _ in range(num_envs)]
    gr00t_img_buffers = [{} for _ in range(num_envs)]

    # initialize buffers
    for env_id in range(num_envs):
        # parquet
        gr00t_data_buffers[env_id] = {
            "observation.state": [],
            "action": [],
            "timestamp": [],
            "annotation.human.action.task_description": [],
            "task_index": [],
            "annotation.human.validity": [],
            "episode_index": [],
            "index": [],
            "next.reward": [],
            "next.done": [],
        }

        # image
        for env_id in range(num_envs):
            gr00t_img_buffers[env_id] = {
                "left_camera": [],
                "right_camera": [],
                "wrist_camera": [],
            }

    eef_position_prev = np.zeros(3)
    eef_orientation_prev = np.array([0.0, 0.0, 0.0, 1.0])
    previous_time = timeline.get_current_time()

    env_id = 0
    task_completed = False

    print("Starting pick-and-place execution")
    while simulation_app.is_running():
        if SimulationManager.is_simulating() and not task_completed:
            if reset_needed:
                random_x = np.random.uniform(0.4, 0.6)
                random_y = np.random.uniform(-0.4, 0.4)

                reset_position = np.array([random_x, random_y, 0.0258])
                
                pick_place.reset(reset_position)

                random_x = np.random.uniform(0.4, 0.6)
                random_y = np.random.uniform(-0.4, 0.4)
                place_position = np.array([random_x, random_y, 0.12])

                pick_place.set_target(place_position)
                reset_needed = False
                
                env_id += 1
                print(env_id)

            if env_id == num_envs:
                task_completed = True

            # Execute one step of the pick-and-place operation
            pick_place.forward(args.ik_method)
            eef_position, eef_orientation = pick_place.end_effector_link.get_world_poses()
            eef_position = np.squeeze(eef_position)
            eef_orientation = np.squeeze(eef_orientation)

            current_time = np.round(timeline.get_current_time(), 3)

            cube_position = pick_place.cube.get_world_poses()[0].numpy()
            true_success = np.linalg.norm(place_position - cube_position) < 0.1
            
            # save prev data
            eef_position_prev = eef_position
            eef_orientation_prev = eef_orientation
            

            
            

        if pick_place.is_done() and not task_completed:
            # print("done picking and placing")
            # task_completed = False
            reset_needed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        simulation_app.close()
