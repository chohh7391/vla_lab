# TODO: `pick_place` Task 구현 계획 (manager-based, RL)

## 목표
- teleoperation/mimic 방식(`stack`)이 아니라 **reward 기반 RL**로 학습하는 pick & place task 신규 구축.
- **manager-based** 아키텍처로 구현 (IsaacLab `Isaac-Lift-Cube-Franka` 환경이 정석 레퍼런스).
- 물체 = **큐브**, 목표 = **goal pose command + 시각화 마커**.
- **full pick & place**: 그리퍼 open/close를 정책이 직접 학습 (reach→grasp→lift→place).
- 로봇 = forge가 쓰는 **FR3** (단, 아래 PD 게인 이슈 주의), 컨트롤러 = forge의 task-space 제어에 가장 가까운 **IK-relative** (+ joint-position 변형).

## 레퍼런스
- 구조/보상 정석: `_isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
  - `lift_env_cfg.py`, `mdp/rewards.py`, `mdp/observations.py`, `mdp/terminations.py`
  - `config/franka/{joint_pos_env_cfg,ik_rel_env_cfg}.py`, `config/franka/agents/rsl_rl_ppo_cfg.py`
- repo 내 manager-based 관례: `source/vla_lab/vla_lab/tasks/manager_based/base_line/stack/`
- 로봇/제어 레퍼런스(direct): `source/vla_lab/vla_lab/tasks/direct/base_line/forge/`, `.../factory/`
- FR3 asset: `source/vla_lab/vla_lab/assets/fr3_with_ft_sensor/fr3_with_ft_sensor.py`

---

## ⚠️ 먼저 해결해야 할 핵심 이슈

### 1) FR3 액추에이터 PD 게인 (블로커)
- `FR3_WITH_FT_SENSOR`는 arm 액추에이터 `stiffness=0, damping=0` → forge의 **토크/OSC 컨트롤러 전용**.
- manager-based의 `JointPositionAction` / `DifferentialInverseKinematicsAction`은 **implicit actuator의 PD로 position target을 추종**하므로, 게인 0이면 **팔이 안 움직임**.
- **해결**: FR3에 high-PD를 입힌 변형 `FR3_WITH_FT_SENSOR_HIGH_PD`를 만든다.
  - arm(`fr3_joint[1-4]`, `fr3_joint5/6/7`): `stiffness=400.0, damping=80.0` (Franka Panda HIGH_PD 기준값, FR3≈Panda이므로 출발점으로 적절. 추후 튜닝).
  - gripper(`fr3_finger_joint[1-2]`): 기존 `stiffness=7500, damping=173` 유지 (binary position 제어에 적합).
  - 구현 시 module-level 싱글톤 오염 방지를 위해 `copy.deepcopy` 후 수정하거나 asset 모듈에 별도 cfg 추가.
- (대안) 리스크 줄이려면 1차 검증은 `FRANKA_PANDA_HIGH_PD_CFG`(stack에서 검증됨)로 돌리고, 이후 FR3로 교체.

### 2) FR3 link/joint 이름 (코드베이스에서 확인된 것)
- joints: `fr3_joint1`~`fr3_joint7`, `fr3_finger_joint1`/`fr3_finger_joint2`
- bodies: `fr3_hand`, `fr3_hand_tcp`, `fr3_leftfinger`, `fr3_rightfinger`
- **미확인**: base link 이름 (FrameTransformer source). Franka 표준상 `fr3_link0`로 추정 → USD에서 반드시 확인 필요.
- EE 프레임은 `fr3_hand_tcp`(확인됨)를 그대로 쓰면 offset 불필요 (factory_env가 이 body를 fingertip으로 사용 중).

---

## 파일 구조 (신규)
```
source/vla_lab/vla_lab/tasks/manager_based/base_line/pick_place/
  __init__.py
  pick_place_env_cfg.py          # Scene + MDP(보상/종료/관측/커맨드/이벤트) base, lift_env_cfg 미러
  mdp/
    __init__.py                  # from isaaclab.envs.mdp import *; from .rewards/observations/terminations import *
    observations.py              # object_position_in_robot_root_frame 등
    rewards.py                   # object_ee_distance, object_is_lifted, object_goal_distance (+ grasp/release)
    terminations.py              # object_dropping, object_reached_goal
  config/
    __init__.py
    franka/
      __init__.py                # gym.register
      joint_pos_env_cfg.py       # FR3 + JointPositionAction + BinaryJointPositionAction
      ik_rel_env_cfg.py          # FR3 HIGH_PD + DifferentialIK(rel) + BinaryJointPositionAction
      agents/
        __init__.py
        rsl_rl_ppo_cfg.py        # LiftCube PPO 러너 미러
        rl_games_ppo_cfg.yaml    # (선택)
```
- 상위 패키지 `__init__.py`들에서 import/등록 누락 없는지 확인.

---

## 구현 단계

### Step 1 — Scene (`pick_place_env_cfg.py: ObjectTableSceneCfg`)
- `robot`(MISSING, config에서 주입), `ee_frame`(MISSING), `object`(MISSING) + table/plane/light.
- lift_env_cfg의 `ObjectTableSceneCfg` 그대로 차용.

### Step 2 — Commands (goal pose)
- `mdp.UniformPoseCommandCfg`로 `object_pose` 정의 (`debug_vis=True` → goal 마커 자동 표시).
- `body_name`은 config에서 `fr3_hand_tcp`로 주입.
- ranges: `pos_x=(0.4,0.6), pos_y=(-0.25,0.25), pos_z=(0.25,0.5)`, roll/pitch/yaw=0 (place 높이 포함).

### Step 3 — Actions (config 별 주입)
- joint_pos: `mdp.JointPositionActionCfg(joint_names=["fr3_joint.*"], scale=0.5, use_default_offset=True)`
- ik_rel: `DifferentialInverseKinematicsActionCfg(joint_names=["fr3_joint.*"], body_name="fr3_hand_tcp", controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), scale=0.5)`
  - `fr3_hand_tcp`를 EE로 쓰면 body_offset 불필요. (`fr3_hand`+offset 방식 쓸 경우 offset≈[0,0,0.107])
- gripper(공통): `mdp.BinaryJointPositionActionCfg(joint_names=["fr3_finger.*"], open=0.04, close=0.0)`
  - → 그리퍼 open/close가 학습 대상(이산). 연속 제어 원하면 별도 action term 검토.

### Step 4 — Observations (`PolicyCfg`)
- `joint_pos_rel`, `joint_vel_rel`, `object_position_in_robot_root_frame`,
  `generated_commands(command_name="object_pose")`, `last_action`.
- `concatenate_terms=True`, `enable_corruption=True`.

### Step 5 — Rewards (`RewardsCfg`) — lift `mdp/rewards.py` 차용 + pick&place 보강
- `reaching_object` = `object_ee_distance(std=0.1)`, weight 1.0  ← reach
- `lifting_object` = `object_is_lifted(minimal_height=0.04)`, weight 15.0  ← grasp/lift 게이트
- `object_goal_tracking` = `object_goal_distance(std=0.3, minimal_height=0.04)`, weight 16.0  ← place (lifted일 때만)
- `object_goal_tracking_fine_grained` = `object_goal_distance(std=0.05, ...)`, weight 5.0
- `action_rate` = `action_rate_l2`, weight -1e-4 (+curriculum로 -1e-1)
- `joint_vel` = `joint_vel_l2`, weight -1e-4 (+curriculum)
- (선택 보강) grasp 보너스: EE-object 근접 + 그리퍼 close 상태 reward term 추가 → pick 안정화.
- (선택) place 후 release 보상: object가 goal 근처 + 그리퍼 open 시 보너스.

### Step 6 — Terminations (`TerminationsCfg`)
- `time_out` (`time_out=True`)
- `object_dropping` = `root_height_below_minimum(minimum_height=-0.05, asset_cfg=object)`
- (선택) success 종료: `object_reached_goal(threshold=0.02)` (terminations.py에 이미 있음, 조기 종료 원하면 사용)

### Step 7 — Events (`EventCfg`)
- `reset_all` = `reset_scene_to_default(mode="reset")`
- `reset_object_position` = `reset_root_state_uniform(pose_range={x:(-0.1,0.1), y:(-0.25,0.25), z:0}, asset_cfg=object)`
- (선택) FR3 초기 관절 랜덤화 (stack의 `franka_stack_events` 참고).

### Step 8 — Curriculum (`CurriculumCfg`)
- `action_rate`, `joint_vel` 가중치를 `num_steps=10000`에서 -1e-1로 강화 (lift 동일).

### Step 9 — Env 조립 (`PickPlaceEnvCfg(ManagerBasedRLEnvCfg)`)
- scene/observations/actions/commands/rewards/terminations/events/curriculum 결합.
- `__post_init__`: `decimation=2`, `episode_length_s=5.0`(place 포함 시 5~10s 검토), `sim.dt=0.01`, physx 설정은 lift 값 차용.

### Step 10 — Config (FR3 주입)
- `joint_pos_env_cfg.py`: `FrankaCubePickPlaceEnvCfg(PickPlaceEnvCfg)`
  - robot = FR3(기본 게인은 joint_pos도 PD 필요 → **HIGH_PD 변형 사용 권장**), object(DexCube), ee_frame(`fr3_hand_tcp`), `commands.object_pose.body_name="fr3_hand_tcp"`.
- `ik_rel_env_cfg.py`: HIGH_PD FR3 + DifferentialIK(rel).
- DexCube: `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd`, `disable_gravity=False`.
- `_PLAY` 변형(num_envs 축소, corruption off)도 추가.

### Step 11 — 등록 (`config/franka/__init__.py`)
- `gym.register("VlaLab-BaseLine-PickPlace-Franka-v0", ...)` (joint_pos)
- `gym.register("VlaLab-BaseLine-PickPlace-Franka-IK-Rel-v0", ...)` (ik_rel)
- `rsl_rl_cfg_entry_point` → `rsl_rl_ppo_cfg:PickPlacePPORunnerCfg` (LiftCube 미러, `experiment_name="fr3_pick_place"`).

### Step 12 — 검증
- import 스모크: 환경 cfg가 에러 없이 생성되는지.
- 짧은 학습 1회: reward 상승 + `lifting_object`/`object_goal_tracking` 증가 곡선 확인.
- FR3 팔이 실제로 움직이는지(=PD 게인 이슈 해결됐는지) 우선 확인.

---

## 미해결/결정 필요
- [ ] FR3 base link 이름 USD에서 확인 (`fr3_link0` 추정) — FrameTransformer/IK source.
- [ ] FR3 HIGH_PD 게인값 튜닝 (Panda 400/80 출발).
- [ ] 1차 검증을 Panda로 먼저 할지, 바로 FR3로 갈지.
- [ ] 그리퍼 제어를 binary 유지할지 연속으로 바꿀지.
- [ ] episode 길이 (place까지 5s vs 10s).
- [ ] grasp 보너스/release 보상 추가 여부.
