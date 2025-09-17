# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .forge_env import ForgeEnv
from .forge_env_cfg import ForgeTaskGearMeshCfg, ForgeTaskNutThreadCfg, ForgeTaskPegInsertCfg

##
# Register Gym environments.
##

gym.register(
    id="VlaLab-BaseLine-Forge-PegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_peg_insert.yaml",
    },
)

gym.register(
    id="VlaLab-BaseLine-Forge-GearMesh-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskGearMeshCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_gear_mesh.yaml",
    },
)

gym.register(
    id="VlaLab-BaseLine-Forge-NutThread-Direct-v0",
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskNutThreadCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
