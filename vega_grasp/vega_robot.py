import os
import torch
import sapien
import numpy as np
import gymnasium as gym
from typing import Dict, cast
from scipy.spatial.transform import Rotation
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class Vega(BaseAgent):
    '''Only enable the right arm and right hand
    '''
    uid = "vega"
    urdf_path = os.path.join(os.path.dirname(__file__), "dexmate-urdf/robots/humanoid/vega_1/vega_upper_body_mod.urdf")
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
            # # Left RGB camera from URDF
            # CameraConfig(
            #     uid="zed_left_camera",
            #     pose=Pose.create_from_pq(
            #         p=[0.0365, 0.023, 0.0489],
            #         q=[1, 0, 0, 0]
            #     ),
            #     width=128,
            #     height=128,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.robot.links_map["head_l3"],
            # ),
            # # Right RGB camera from URDF
            # CameraConfig(
            #     uid="zed_right_camera",
            #     pose=Pose.create_from_pq(
            #         p=[0.0365, -0.027, 0.0489],
            #         q=[1, 0, 0, 0]
            #     ),
            #     width=128,
            #     height=128,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.robot.links_map["head_l3"],
            # ),
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


    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "R_hand_base"
        )

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., [2, 5, 8, 10, 12, 14, 16]]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    def is_grasping(self, object: Actor, min_force=0.5, max_distance=0.1, max_relative_velocity=0.5, max_angle=85):
        """Check if the robot is grasping an object
        
        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_distance (float, optional): Maximum relative distance to consider a link as candidate. Defaults to 0.1.
            max_relative_velocity (float, optional): Maximum relative velocity to consider a link as candidate. Defaults to 0.5.
            max_angle (int, optional): Maximum angle between the two force directions (degrees). Defaults to 85.
        """
        # 定义链接分组
        thumb_link_names = ['R_th_l0', 'R_th_l1', 'R_th_l2', 'R_th_tip']
        finger_link_names = ['R_ff_l1', 'R_mf_l1', 'R_rf_l1', 'R_lf_l1', 
                            'R_ff_l2', 'R_mf_l2', 'R_rf_l2', 'R_lf_l2', 
                            'R_ff_tip', 'R_mf_tip', 'R_rf_tip', 'R_lf_tip']
        
        # 获取所有链接及其名称
        all_link_names = thumb_link_names + finger_link_names
        link_dict = {name: self.robot.links_map[name] for name in all_link_names if name in self.robot.links_map}
        
        # 获取物体的位置和速度
        object_pos = object.pose.p  # (batch_size, 3)
        object_vel = object.linear_velocity  # (batch_size, 3)
        
        # 第一步：用相对距离和相对速度过滤候选链接
        candidate_links = []
        for link_name, link in link_dict.items():
            link_pos = link.pose.p  # (batch_size, 3)
            link_vel = link.linear_velocity  # (batch_size, 3)
            
            # 计算相对距离
            relative_distance = torch.linalg.norm(link_pos - object_pos, dim=1)  # (batch_size,)
            
            # 计算相对速度
            relative_velocity = torch.linalg.norm(link_vel - object_vel, dim=1)  # (batch_size,)
            
            # 过滤条件：距离小于阈值且相对速度小于阈值
            is_candidate = torch.logical_and(
                relative_distance <= max_distance,
                relative_velocity <= max_relative_velocity
            )
            
            # 如果至少有一个batch满足条件，则加入候选列表
            if torch.any(is_candidate):
                candidate_links.append((link_name, link, is_candidate))
        
        if len(candidate_links) == 0:
            # 如果没有候选链接，返回False
            batch_size = object_pos.shape[0]
            return torch.zeros(batch_size, dtype=torch.bool, device=object_pos.device)
        
        # 第二步：分组并计算有效接触的链接
        thumb_links = []
        finger_links = []
        
        for link_name, link, is_candidate in candidate_links:
            if link_name in thumb_link_names:
                thumb_links.append((link, is_candidate))
            elif link_name in finger_link_names:
                finger_links.append((link, is_candidate))
        
        # 第三步：计算有效接触的链接（力的大小大于min_force）
        thumb_contact_forces = []
        finger_contact_forces = []
        
        batch_size = object_pos.shape[0]
        device = object_pos.device
        
        for link, is_candidate in thumb_links:
            contact_forces = self.scene.get_pairwise_contact_forces(link, object)  # (batch_size, 3)
            force_magnitude = torch.linalg.norm(contact_forces, dim=1)  # (batch_size,)
            
            # 只在候选链接中且力大于阈值时考虑
            valid_contact = torch.logical_and(is_candidate, force_magnitude >= min_force)
            
            # 只保留有效接触的力
            valid_forces = contact_forces * valid_contact.unsqueeze(1).float()
            thumb_contact_forces.append(valid_forces)
        
        for link, is_candidate in finger_links:
            contact_forces = self.scene.get_pairwise_contact_forces(link, object)  # (batch_size, 3)
            force_magnitude = torch.linalg.norm(contact_forces, dim=1)  # (batch_size,)
            
            # 只在候选链接中且力大于阈值时考虑
            valid_contact = torch.logical_and(is_candidate, force_magnitude >= min_force)
            
            # 只保留有效接触的力
            valid_forces = contact_forces * valid_contact.unsqueeze(1).float()
            finger_contact_forces.append(valid_forces)
        
        # 如果两组都没有有效接触，返回False
        if len(thumb_contact_forces) == 0 or len(finger_contact_forces) == 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 第四步：计算两组的总力方向，检查是否相对
        # 计算总力
        thumb_total_force = torch.stack(thumb_contact_forces, dim=0).sum(dim=0)  # (batch_size, 3)
        finger_total_force = torch.stack(finger_contact_forces, dim=0).sum(dim=0)  # (batch_size, 3)
        
        # 计算总力的大小
        thumb_force_magnitude = torch.linalg.norm(thumb_total_force, dim=1)  # (batch_size,)
        finger_force_magnitude = torch.linalg.norm(finger_total_force, dim=1)  # (batch_size,)
        
        # 检查两组都有足够的力
        has_thumb_force = thumb_force_magnitude >= min_force
        has_finger_force = finger_force_magnitude >= min_force
        
        # 计算两组力方向之间的角度（只在两组都有力时计算）
        # 对于没有足够力的情况，角度设为0（这样is_opposing会是False）
        angle = torch.zeros(batch_size, device=device)
        valid_mask = torch.logical_and(has_thumb_force, has_finger_force)
        if torch.any(valid_mask):
            angle[valid_mask] = common.compute_angle_between(
                thumb_total_force[valid_mask], 
                finger_total_force[valid_mask]
            )
        
        angle_deg = torch.rad2deg(angle)  # (batch_size,)
        
        # 检查角度是否接近180度（方向相对）
        # 角度应该在 (180 - max_angle) 到 (180 + max_angle) 之间，但由于角度范围是0-180，我们检查是否 >= (180 - max_angle)
        is_opposing = angle_deg >= (180 - max_angle)
        
        # 最终判断：两组都有足够的力，且方向相对
        result = torch.logical_and(
            torch.logical_and(has_thumb_force, has_finger_force),
            is_opposing
        )
        
        print(f"Get {torch.sum(result)} valid grasp in {batch_size} candidates")
        return result


if __name__ == "__main__":
    # from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
    from pick_single_ycb import PickSingleYCBEnv
    # env = gym.make("Empty-v1", robot_uids="vega")
    env = gym.make("MyPickSingleYCB-v1", robot_uids="vega")
