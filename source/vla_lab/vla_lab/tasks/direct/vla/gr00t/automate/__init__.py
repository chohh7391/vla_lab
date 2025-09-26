# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

################## GR00T Env ##################
gym.register(
    id="VlaLab-Gr00t-AutoMate-Assembly-Direct-v1",
    entry_point=f"{__name__}.assembly_gr00t_env:AssemblyGr00tEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assembly_gr00t_env_cfg:AutomateTaskAssemblyGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


################## GR00T Demo Save Env ##################
gym.register(
    id="VlaLab-Gr00t-AutoMate-Assembly-Demo-Save-Direct-v0",
    entry_point=f"{__name__}.assembly_gr00t_demo_save_env:AssemblyGr00tDemoSaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assembly_gr00t_env_cfg:AutomateTaskAssemblyGr00tDemoSaveCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


################## GR00T Old Env ##################
# Not parallel env
# just for comparing with old results
gym.register(
    id="VlaLab-Gr00t-AutoMate-Disassembly-Direct-v0",
    entry_point=f"{__name__}.assembly_gr00t_not_parallel_env:AssemblyGr00tNotParallelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assembly_gr00t_env_cfg:AutomateTaskAssemblyGr00tCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)