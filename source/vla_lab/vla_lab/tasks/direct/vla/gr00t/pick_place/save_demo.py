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


def save_gr00t_observations(
    dataset_path: str, chunk_id: str, gr00t_data_buffers, gr00t_img_buffers, num_envs
):
    data_dir = os.path.join(dataset_path, "data", chunk_id)
    video_dir = {
        "left_camera": os.path.join(dataset_path, "videos", chunk_id, "observation.images.left_view"),
        "right_camera": os.path.join(dataset_path, "videos", chunk_id, "observation.images.right_view"),
        "wrist_camera": os.path.join(dataset_path, "videos", chunk_id, "observation.images.wrist_view"),
    }

    print(f"Saving data to {data_dir}...")

    for env_id in range(num_envs):
        # Save Parquet Data
        # 데이터가 비어있지 않은지 확인
        if len(gr00t_data_buffers[env_id]["observation.state"]) == 0:
            print(f"Warning: Buffer for env_id {env_id} is empty. Skipping.")
            continue

        df = pd.DataFrame(gr00t_data_buffers[env_id])
        gr00t_data_path = os.path.join(data_dir, f"episode_{env_id:06d}.parquet")
        os.makedirs(os.path.dirname(gr00t_data_path), exist_ok=True)
        df.to_parquet(gr00t_data_path, index=False)

        # Save Videos
        for camera_name, img_list in gr00t_img_buffers[env_id].items():

            # (T, H, W, C) 형태로 저장되어 있으므로 그대로 stack
            img_tensor = torch.stack(img_list, dim=0).to(device="cuda:0", dtype=torch.uint8)
            
            gr00t_video_path = os.path.join(video_dir[camera_name], f"episode_{env_id:06d}.mp4")
            os.makedirs(os.path.dirname(gr00t_video_path), exist_ok=True)
            
            # write_video expects (T, H, W, C) for uint8 tensor
            write_video(gr00t_video_path, img_tensor, fps=60, video_codec="h264")

    print(f"Saved {num_envs} episodes.")


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
    
    # 버퍼 초기화
    gr00t_data_buffers = [
        {
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
        } for _ in range(num_envs)
    ]

    gr00t_img_buffers = [
        {
            "left_camera": [],
            "right_camera": [],
            "wrist_camera": [],
        } for _ in range(num_envs)
    ]

    # 이전 상태 저장 변수 초기화
    eef_position_prev = np.zeros(3)
    eef_orientation_prev = np.array([0.0, 0.0, 0.0, 1.0])
    gripper_qpos_prev = np.array([0])

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
                save_gr00t_observations(
                    dataset_path="/home/home/datasets/gr00t-rl/pick_place",
                    chunk_id="chunk-000",
                    gr00t_data_buffers=gr00t_data_buffers,
                    gr00t_img_buffers=gr00t_img_buffers,
                    num_envs=num_envs,
                )
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
            
            current_time = np.round(timeline.get_current_time(), 3)
            cube_position = pick_place.cube.get_world_poses()[0].numpy()
            true_success = np.linalg.norm(place_position - cube_position) < 0.1
            
            # --- Get Images ---
            # None 체크 및 복사
            left_rgb = pick_place.left_camera.get_rgb()
            left_cam_data = left_rgb.copy() if left_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            
            right_rgb = pick_place.right_camera.get_rgb()
            right_cam_data = right_rgb.copy() if right_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)
            
            wrist_rgb = pick_place.wrist_camera.get_rgb()
            wrist_cam_data = wrist_rgb.copy() if wrist_rgb is not None else np.zeros((256, 256, 3), dtype=np.uint8)

            # --- Calculate Actions (Delta) ---
            delta_pos_world = eef_position - eef_position_prev
            action_pos = delta_pos_world / 0.2

            # [중요] Isaac Sim(w,x,y,z) -> SciPy(x,y,z,w) 순서 변환
            # eef_orientation이 [w, x, y, z]라고 가정 시:
            quat_curr_scipy = eef_orientation[[1, 2, 3, 0]]
            quat_prev_scipy = eef_orientation_prev[[1, 2, 3, 0]]

            current_quat_obj = R.from_quat(quat_curr_scipy)      
            prev_quat_obj = R.from_quat(quat_prev_scipy)    

            # World Frame 회전 차이 (Q_diff * Q_prev = Q_curr)
            diff_quat_obj = current_quat_obj * prev_quat_obj.inv()

            delta_rot_vec = diff_quat_obj.as_rotvec()
            action_rot = delta_rot_vec / 0.097

            state = np.concatenate((eef_position, eef_orientation, gripper_qpos), axis=-1)
            arm_action = np.concatenate((action_pos, action_rot), axis=-1)
            gripper_action = gripper_qpos_prev
            prev_action = np.concatenate((arm_action, gripper_action), axis=-1)

            is_done = pick_place.is_done()

            # --- Data Appending ---
            # 현재 env_id 버퍼에 저장
            if env_id < num_envs:
                buf = gr00t_data_buffers[env_id]
                img_buf = gr00t_img_buffers[env_id]

                buf["observation.state"].append(state.tolist())
                buf["action"].append(prev_action.tolist())
                buf["timestamp"].append(current_time)
                buf["annotation.human.action.task_description"].append(0)
                buf["task_index"].append(0)
                buf["annotation.human.validity"].append(true_success)
                buf["episode_index"].append(env_id)
                buf["index"].append(total_step)
                buf["next.reward"].append(1.0 if true_success else 0.0)
                buf["next.done"].append(is_done)

                # 이미지 저장 (H, W, C) 형태 그대로 유지 -> 나중에 write_video가 처리
                img_buf["left_camera"].append(torch.from_numpy(left_cam_data))
                img_buf["right_camera"].append(torch.from_numpy(right_cam_data))
                img_buf["wrist_camera"].append(torch.from_numpy(wrist_cam_data))

            # --- Update Prev State ---
            eef_position_prev = eef_position
            eef_orientation_prev = eef_orientation
            gripper_qpos_prev = gripper_qpos
            
            total_step += 1
            episode_step += 1
            
            # --- Check Episode End ---
            if is_done:
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