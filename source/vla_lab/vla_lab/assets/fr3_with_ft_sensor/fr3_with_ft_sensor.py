import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import numpy as np

from .. import ASSET_DIR


FR3_WITH_FT_SENSOR = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/fr3_with_ft_sensor/fr3_with_ft_sensor.usd",
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": float(-np.pi/4),
            "fr3_joint3": 0.0,
            "fr3_joint4": float(-3*np.pi/4),
            "fr3_joint5": 0.0,
            "fr3_joint6": float(np.pi/2),
            "fr3_joint7": float(np.pi/4),
            "fr3_finger_joint1": 0.04,  # opened
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "fr3_arm1": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87.0,
            velocity_limit_sim=2.62,
        ),
        "fr3_joint5": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint5"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=5.26,
        ),
        "fr3_joint6": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint6"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=4.18,
        ),
        "fr3_joint7": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint7"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=5.26,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint[1-2]"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.04,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)