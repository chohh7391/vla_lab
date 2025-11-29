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
parser.add_argument("--headless", action="store_true", help="headless mode")
parser.add_argument("--num_envs", type=int, default=4, help="number of demos")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# Headless 모드 설정 (필요시 True로 변경)
simulation_app = SimulationApp({"headless": args.headless})

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from pick_place import FrankaPickPlace
import numpy as np
from scipy.spatial.transform import Rotation as R

import pandas as pd
import sys
import os
from torchvision.io import write_video
import torch

from vla_lab.envs.utils.gr00t_service import AsyncExternalRobotInferenceClient

def get_gr00t_observation(
    left_cam_data, right_cam_data, wrist_cam_data, eef_position, eef_orientation, gripper_qpos
):

    observations = {
        "video.left_view": left_cam_data.astype(np.uint8),
        "video.right_view": right_cam_data.astype(np.uint8),
        "video.wrist_view": wrist_cam_data.astype(np.uint8),
        "state.eef_position": eef_position.astype(np.float64),
        "state.eef_quaternion": eef_orientation.astype(np.float64),
        "state.gripper_qpos": gripper_qpos.astype(np.float64),
    }
    return observations


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

    num_envs = args.num_envs

    gr00t_chunk_size = 16
    gr00t_policy = AsyncExternalRobotInferenceClient(host="localhost", port=5555)
    
    env_id = 0
    total_step = 0
    episode_step = 0
    
    reset_needed = True
    task_completed = False

    print("Starting pick-and-place execution")
    
    while simulation_app.is_running():
        # [수정] 모든 에피소드 완료 시 저장 후 종료
        if env_id >= num_envs:
            if not task_completed:
                task_completed = True
            break

        if SimulationManager.is_simulating():
            # [수정] 리셋 로직
            if reset_needed:
                print(f"--- Starting Episode {env_id} ---")
                timeline.set_current_time(0.0)
                
                # 1. 로봇 및 타겟 위치 리셋
                random_x = np.random.uniform(0.4, 0.6)
                random_y = np.random.uniform(-0.4, 0.4)
                reset_position = np.array([random_x, random_y, 0.0258])
                pick_place.reset(reset_position)

                random_x_target = np.random.uniform(0.4, 0.6)
                random_y_target = np.random.uniform(-0.4, 0.4)
                place_position = np.array([random_x_target, random_y_target, 0.12])
                pick_place.set_target(place_position)
                
                # 2. 물리 엔진 업데이트 (위치 적용을 위해)
                simulation_app.update()
                
                # 3. [중요] 이전 상태 변수들을 현재 리셋된 상태로 갱신 (Action 튀는 것 방지)
                eef_pos_reset, eef_quat_reset = pick_place.end_effector_link.get_world_poses()
                eef_position_prev = np.squeeze(eef_pos_reset)
                eef_orientation_prev = np.squeeze(eef_quat_reset)
                gripper_qpos_prev = pick_place.robot.get_dof_positions(dof_indices=[7, 8])[0]
                
                reset_needed = False
                episode_step = 0
                
                # 리셋 직후 프레임은 건너뛰고 다음 프레임부터 기록
                continue

            # --- Control Step ---
            pick_place.forward(args.ik_method)
            
            # --- Get Current State ---
            eef_position, eef_orientation = pick_place.end_effector_link.get_world_poses()
            eef_position = np.squeeze(eef_position)
            eef_orientation = np.squeeze(eef_orientation)
            gripper_qpos = pick_place.robot.get_dof_positions(dof_indices=[7, 8])[0]
            
            left_rgb = pick_place.left_camera.get_rgb()
            left_cam_data = left_rgb.copy() if left_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            
            right_rgb = pick_place.right_camera.get_rgb()
            right_cam_data = right_rgb.copy() if right_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            
            wrist_rgb = pick_place.wrist_camera.get_rgb()
            wrist_cam_data = wrist_rgb.copy() if wrist_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)

            gr00t_observation = get_gr00t_observation(
                left_cam_data, right_cam_data, wrist_cam_data, eef_position, eef_orientation, gripper_qpos
            )

            if total_step % gr00t_chunk_size == 0:

                gr00t_action = gr00t_policy.get_action_sync(gr00t_observation)

            #######################
            
            total_step += 1
            episode_step += 1
            
            # --- Check Episode End ---
            if pick_place.is_done():
                print(f"Episode {env_id} Completed.")
                env_id += 1        # 다음 에피소드로 인덱스 변경
                reset_needed = True # 리셋 플래그 설정 (다음 루프에서 리셋 실행)

        simulation_app.update()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()