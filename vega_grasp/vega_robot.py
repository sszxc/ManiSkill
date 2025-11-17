import sapien
import numpy as np
import gymnasium as gym
from typing import Dict, cast
from scipy.spatial.transform import Rotation
import mani_skill.envs
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose


@register_agent()
class Vega(BaseAgent):
    '''Only enable the right arm and right hand
    '''
    uid = "vega"
    urdf_path = "dexmate-urdf/robots/humanoid/vega_1/vega_upper_body_mod.urdf"
    disable_self_collisions = True

    # ---------- keyframes ---------- #
    _rest_qpos = np.zeros(39)
    _rest_qpos[0] = 0.35  # 低头角度
    _rest_qpos[1] = 1.57
    _rest_qpos[4] = 1.55
    _rest_qpos[7] = 1.57
    _rest_qpos[9] = -2.2
    _rest_qpos[13] = -0.75
    keyframes = dict(
        rest=Keyframe(
            qpos=_rest_qpos,
            pose=sapien.Pose(),
        )
    )

    # ---------- control ---------- #
    arm_joint_names = [
        "R_arm_j1",
        "R_arm_j2",
        "R_arm_j3",
        "R_arm_j4",
        "R_arm_j5",
        "R_arm_j6",
        "R_arm_j7",
    ]
    hand_th_j0_joint_names = [
        "R_th_j0",
    ]
    hand_mimic_joint_names = [
        "R_ff_j1",
        "R_mf_j1",
        "R_rf_j1",
        "R_lf_j1",
        "R_th_j1",
        "R_ff_j2",
        "R_mf_j2",
        "R_rf_j2",
        "R_lf_j2",
        "R_th_j2",
    ]
    hand_mimic: Dict[str, dict[str, float]] = cast(
        Dict[str, dict[str, float]],
        {
            "R_ff_j2": dict(joint="R_ff_j1", multiplier=1.13028, offset=-0.00053),
            "R_mf_j2": dict(joint="R_mf_j1", multiplier=1.13311, offset=-0.00079),
            "R_rf_j2": dict(joint="R_rf_j1", multiplier=1.12935, offset=0.00065),
            "R_lf_j2": dict(joint="R_lf_j1", multiplier=1.15037, offset=0.00186),
            "R_th_j2": dict(joint="R_th_j1", multiplier=1.35316, offset=0.00765),
        },
    )

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    hand_stiffness = 1e3
    hand_damping = 1e2
    hand_force_limit = 100
    
    passive_joint_names = [  
        'head_j1',
        'L_arm_j1',
        # 'R_arm_j1',
        'head_j2',
        'L_arm_j2',
        # 'R_arm_j2',
        'head_j3',
        'L_arm_j3',
        # 'R_arm_j3',
        'L_arm_j4',
        # 'R_arm_j4',
        'L_arm_j5',
        # 'R_arm_j5',
        'L_arm_j6',
        # 'R_arm_j6',
        'L_arm_j7',
        # 'R_arm_j7',
        'L_th_j0',
        'L_ff_j1',
        'L_mf_j1',
        'L_rf_j1',
        'L_lf_j1',
        # 'R_th_j0',
        # 'R_ff_j1',
        # 'R_mf_j1',
        # 'R_rf_j1',
        # 'R_lf_j1',
        'L_th_j1',
        'L_ff_j2',
        'L_mf_j2',
        'L_rf_j2',
        'L_lf_j2',
        # 'R_th_j1',
        # 'R_ff_j2',
        # 'R_mf_j2',
        # 'R_rf_j2',
        # 'R_lf_j2',
        'L_th_j2',
        # 'R_th_j2'
    ]
    passive_joints = PassiveControllerConfig(  
        joint_names=passive_joint_names,
        damping=0,  
        friction=0,  
    )

    @property
    def _controller_configs(self):
        # pod
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        hand_th_j0_pd_joint_pos = PDJointPosControllerConfig(
            self.hand_th_j0_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
        )
        hand_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.hand_mimic_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            mimic=self.hand_mimic,
        )
        # delta pos
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.5,
            upper=0.5,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        hand_th_j0_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.hand_th_j0_joint_names,
            lower=-0.2,
            upper=0.2,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
        )
        hand_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.hand_mimic_joint_names,
            lower=-0.2,
            upper=0.2,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            use_delta=True,
            mimic=self.hand_mimic,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
              arm=arm_pd_joint_pos,
              hand_th_j0=hand_th_j0_pd_joint_pos,
              hand_mimic=hand_mimic_pd_joint_pos,
              passive_joints=self.passive_joints,
            ),
            pd_joint_delta_pos=dict(
              arm=arm_pd_joint_delta_pos,
              hand_th_j0=hand_th_j0_pd_joint_delta_pos,
              hand_mimic=hand_mimic_pd_joint_delta_pos,
              passive_joints=self.passive_joints,
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)


    @property
    def _sensor_configs(self):
        # zed_depth_frame: xyz="0.025 0.023 0.0489" rpy="-1.57079 0 -1.57079"
        # zed_left_camera: xyz="0.0365 0.023 0.0489" rpy="-1.57079 0 -1.57079"
        # zed_right_camera: xyz="0.0365 -0.027 0.0489" rpy="-1.57079 0 -1.57079"

        return [
            # Depth camera from URDF
            CameraConfig(
                uid="zed_depth_camera",
                pose=Pose.create_from_pq(
                    p=[0.025, 0.023, 0.0489],
                    q=[1, 0, 0, 0]
                ),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["head_l3"],
            ),
            # Left RGB camera from URDF
            CameraConfig(
                uid="zed_left_camera",
                pose=Pose.create_from_pq(
                    p=[0.0365, 0.023, 0.0489],
                    q=[1, 0, 0, 0]
                ),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["head_l3"],
            ),
            # Right RGB camera from URDF
            CameraConfig(
                uid="zed_right_camera",
                pose=Pose.create_from_pq(
                    p=[0.0365, -0.027, 0.0489],
                    q=[1, 0, 0, 0]
                ),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["head_l3"],
            ),
        ]

    # 需要设置摩擦力的链接
    _contact_link_names = ['R_hand_base', 'R_th_l0', 'R_ff_l1', 'R_mf_l1', 'R_rf_l1', 'R_lf_l1', 'R_th_l1', 'R_ff_l2', 'R_mf_l2', 'R_rf_l2', 'R_lf_l2', 'R_th_l2', 'R_ff_tip', 'R_mf_tip', 'R_rf_tip', 'R_lf_tip', 'R_th_tip']

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            **{link_name: dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ) for link_name in _contact_link_names}
        ),
    )


if __name__ == "__main__":
    env = gym.make("PickSingleYCB-v1", robot_uids="vega")
