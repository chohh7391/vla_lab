import gymnasium as gym

from .franka_stack_ik_rel_gr00t_mimic_env_cfg import FrankaCubeStackIKRelGr00tMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="VlaLab-BaseLine-Stack-IK-Rel-Gr00t-Mimic-v0",
    entry_point=f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_gr00t_mimic_env_cfg.FrankaCubeStackIKRelGr00tMimicEnvCfg,
    },
    disable_env_checker=True,
)