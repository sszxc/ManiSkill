from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from vega_robot import Vega

WARNED_ONCE = False


@register_env("MyPickSingleYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickSingleYCBEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

    **Randomizations:**
    - the object's xy position is randomized on top of a table. It is placed flat on the table
    - the object's z-axis rotation is randomized
    - the object geometry is randomized by randomly sampling any YCB object. (during reconfiguration)

    **Success Conditions:**
    - the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)

    **Goal Specification:**
    - 3D goal position (also visualized in human renders)

    **Additional Notes**
    - On GPU simulation, in order to collect data from every possible object in the YCB database we recommend using at least 128 parallel environments or more, otherwise you will need to reconfigure in order to sample new objects.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickSingleYCB-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch", "vega"]
    agent: Union[Panda, PandaWristCam, Fetch, Vega]
    goal_thresh = 0.025  # threshold for the success distance between the object and the goal position (meters)

    def __init__(
        self,
        *args,
        robot_uids="vega",  # "panda_wristcam",
        robot_init_qpos_noise=0.02,  # standard deviation of the noise added to the robot's initial joint positions
        num_envs=1,  # number of parallel environments
        reconfiguration_freq=None,  # frequency of reconfiguration, None means auto-set based on num_envs
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        # option A: load all available YCB model IDs from the JSON file, exclude hard to grasp objects
        # self.all_model_ids = np.array(
        #     [
        #         k
        #         for k in load_json(
        #             ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
        #         ).keys()
        #         if k
        #         not in [
        #             "022_windex_bottle",
        #             "028_skillet_lid",
        #             "029_plate",
        #             "059_chain",
        #         ]  # NOTE (arth): ignore these non-graspable/hard to grasp ycb objects
        #     ]
        # )
        # option B: manually specify, check mani_skill_data/data/assets/mani_skill2_ycb/info_pick_v0.json
        self.all_model_ids = np.array(
            [
                "003_cracker_box",
                "004_sugar_box",
                "008_pudding_box",
                "009_gelatin_box",
                "026_sponge",
            ]
        )
        # automatically set the reconfiguration frequency based on the number of environments
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        # """return a default camera"""
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        # return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
        """call the camera implemented by the robot"""
        return self.agent._sensor_configs[0]

    @property
    def _default_human_render_camera_configs(self):
        """return the camera configurations for human rendering(higher resolution)"""
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.8, 0, 0]))

    def _load_scene(self, options: dict):
        """load the scene, including the table, YCB objects and the goal position marker"""
        global WARNED_ONCE
        # build the table scene
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build(random_height=True)

        # randomly sample model IDs from the YCB dataset
        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        # if the number of parallel environments is less than the number of available models and not reconfiguring, give a warning
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        # create YCB objects for each environment
        self._objs: list[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

        # create the goal position marker (green sphere)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],  # 绿色
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        """
        calculate the bottom Z coordinate offset for each object, for placing the object bottom at z=0
        """
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        initialize each episode, randomize the object position, goal position and robot initial pose
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # randomize the object position
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.2 - 0.3  # X: [-0.3, -0.1]
            xyz[:, 1] = torch.rand((b,)) * 0.2 - 0.1  # Y: [-0.1, 0.1]
            xyz[:, 2] = self.object_zs[env_idx]       # Z: object bottom height
            # randomize the object rotation around Z axis
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # randomize the goal position
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.2 - 0.3       # X: [-0.3, -0.1]
            goal_xyz[:, 1] = torch.rand((b,)) * 0.2 - 0.1       # Y: [-0.1, 0.1]
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]  # Z: [object height, object height+0.3]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "vega":
                qpos = Vega.keyframes['rest'].qpos.copy()
                noise_indices = [2, 5, 8, 10, 12, 14, 16]  # R_arm joints
                qpos[noise_indices] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(noise_indices)
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.8, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        """
        evaluate the current state, check if the task is successful
        
        Returns:
            dict:
                - is_grasped: whether the object is grasped
                - obj_to_goal_pos: the vector from the object to the goal position
                - is_obj_placed: whether the object is placed in the goal position (within the threshold)
                - is_robot_static: whether the robot is static
                - success: whether the task is successful
        """
        # calculate the vector from the object to the goal position
        obj_to_goal_pos = self.goal_site.pose.p - self.obj.pose.p
        # check if the Euclidean distance < threshold
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        # check if the object is grasped
        is_grasped = self.agent.is_grasping(self.obj)
        # check if the robot is static (joint velocity < 0.2)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            # success condition: object in the goal position and robot static
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )

    def _get_obs_extra(self, info: dict):
        """
        get additional observation information
        """
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,  # TCP pose
            goal_pos=self.goal_site.pose.p,  # goal position
            is_grasped=info["is_grasped"],  # whether the object is grasped
        )
        # if the observation mode contains "state", add more state information
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,  # TCP to goal position
                obj_pose=self.obj.pose.raw_pose,  # object pose
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,  # TCP to object
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,  # object to goal position
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """
        compute the dense reward
        
        rewards:
        r1. reaching reward: encourage the TCP to approach the object
        r2. grasping reward: give reward when the object is grasped
        r3. placing reward: encourage the object to be moved to the goal position after being grasped
        r4. placing success reward: give reward when the object is placed in the goal position
        r5. static reward: give reward when the object is in the goal position and the robot is static
        r6. task success reward: give reward when the task is successful

        penalties:
        p1. object falling: give penalty when the object falls
        p2. arm collision: give penalty when the arm collides with the object
        
        Args:
            obs: observation
            action: action
            info: contains evaluation information
            
        Returns:
            torch.Tensor: reward values
        """
        # r1. reaching reward
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # r2. grasping reward
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # r3. placing reward
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped  # only calculate placing reward when the object is grasped

        # r4. placing success reward
        reward += info["is_obj_placed"] * is_grasped

        # r5. static reward
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        # r6. task success reward
        reward[info["success"]] = 6
        
        # p1.a object falling penalty
        # object_z = self.obj.pose.p[:, 2]
        # table_z = self.table_scene.table_height
        # object_z_below_table_z = object_z < table_z
        # reward -= object_z_below_table_z.float() * 1.0
        # # print(f"detected {torch.sum(object_z_below_table_z)} of {object_z.shape[0]} objects falling")

        # p1.b object away from goal penalty
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        reward -= (obj_to_goal_dist > 0.5).float() * 1.0
        # print(f"detected {torch.sum(obj_to_goal_dist > 0.5)} of {obj_to_goal_dist.shape[0]} objects too far from goal")
        
        # p2. arm collision penalty
        if self.robot_uids == "vega":
            arm_link_names = [
                'R_arm_l1', 'R_arm_l2', 'R_arm_l3', 'R_arm_l4', 'R_arm_l5', 'R_arm_l6', 'R_arm_l7', 'R_arm_l8',
            ]
            
            # check the contact force between each arm link and the object
            arm_collision_penalty = torch.zeros(len(reward), device=reward.device)
            min_contact_force = 0.5  # minimum contact force threshold (Newton)
            
            for link_name in arm_link_names:
                if link_name in self.agent.robot.links_map:
                    link = self.agent.robot.links_map[link_name]
                    # get the contact force between the link and the object
                    contact_forces = self.scene.get_pairwise_contact_forces(link, self.obj)
                    force_magnitude = torch.linalg.norm(contact_forces, dim=1)  # calculate the force magnitude
                    has_collision = force_magnitude >= min_contact_force
                    arm_collision_penalty += has_collision.float() * 1.0  # a fixed penalty
            reward -= arm_collision_penalty
            # print(f"detected {torch.sum(arm_collision_penalty)} of {object_z.shape[0]} objects colliding with arms")

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        """
        compute the normalized dense reward (normalize the reward to the range [0, 1])
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
