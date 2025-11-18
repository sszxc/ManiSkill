"""
PPO algorithm implementation for continuous action space tasks with RGB image observations
Using ManiSkill environment for robot manipulation tasks reinforcement learning training
Reference:
https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo_rgb.py
"""
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from vega_robot import Vega
from pick_single_ycb import PickSingleYCBEnv

@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True  # toggle torch.backends.cudnn.deterministic=False
    cuda: bool = True
    track: bool = False  # wandb tracking
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "PPO"
    capture_video: bool = True  # capture videos of the agent performances (check out `videos` folder)
    save_model: bool = True  # save model into the `runs/{run_name}` folder
    evaluate: bool = False  # only runs evaluation with the given model checkpoint
    checkpoint: Optional[str] = None  # path to a pretrained checkpoint
    render_mode: str = "all"

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    include_state: bool = True  # whether to include state information in observations
    total_timesteps: int = 10000000  # total timesteps of the experiments
    learning_rate: float = 3e-4
    num_envs: int = 512  # the number of parallel environments
    num_eval_envs: int = 8  # the number of parallel evaluation environments
    partial_reset: bool = True  # whether to let parallel environments reset upon termination instead of truncation
    eval_partial_reset: bool = False  # whether to let parallel evaluation environments reset upon termination instead of truncation
    num_steps: int = 50  # the number of steps to run in each environment per policy rollout
    num_eval_steps: int = 50  # the number of steps to run in each evaluation environment during evaluation
    reconfiguration_freq: Optional[int] = None  # how often to reconfigure the environment during training
    eval_reconfiguration_freq: Optional[int] = 1  # for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks
    control_mode: Optional[str] = "pd_joint_delta_pos"  # the control mode to use for the environment
    anneal_lr: bool = False  # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.8  # the discount factor gamma
    gae_lambda: float = 0.9  # the lambda for the general advantage estimation
    num_minibatches: int = 32  # the number of mini-batches
    update_epochs: int = 4  # the K epochs to update the policy
    norm_adv: bool = True  # Toggles advantages normalization
    clip_coef: float = 0.2  # the surrogate clipping coefficient
    clip_vloss: bool = False  # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    ent_coef: float = 0.0  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 0.5  # the maximum norm for the gradient clipping
    target_kl: float = 0.2  # the target KL divergence threshold
    reward_scale: float = 1.0  # Scale the reward by this factor
    eval_freq: int = 25  # evaluation frequency in terms of iterations
    save_train_video_freq: Optional[int] = None  # frequency to save training videos in terms of iterations
    finite_horizon_gae: bool = False  # whether to use finite horizon GAE (Generalized Advantage Estimation)

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    initialize the neural network layer using orthogonal initialization
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    """
    a buffer class for storing dictionary type observation data
    """
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    """
    a feature extraction network for processing multi-modal observations (RGB images and state vectors)
    using Nature CNN architecture to process images, and using linear layers to process state information
    """
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        """
        forward pass: process multi-modal observations and return concatenated feature vector
        """
        encoded_tensor_list = []
        # use different extractors to process each modality of observations
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                # convert RGB image from (B, H, W, C) to (B, C, H, W) and normalize to [0, 1]
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    """
    PPO agent network, containing:
    - feature extraction network (NatureCNN): process multi-modal observations
    - critic network: estimate state value function V(s)
    - actor network: output action mean and standard deviation (for continuous action space)
    """
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        # critic network: output state value function V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        # actor network: output action mean
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        # actor logstd: learnable parameter for action standard deviation, initialized to -0.5
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)

    def get_features(self, x):
        """extract feature representation of observations"""
        return self.feature_net(x)
    
    def get_value(self, x):
        """get state value function V(s)"""
        x = self.feature_net(x)
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        """
        calculate action from observations
        
        Args:
            x: observations
            deterministic: whether to use deterministic policy (return mean) or stochastic policy (sample from distribution)
        """
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        # sample action from normal distribution
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_and_value(self, x, action=None):
        """
        get action, log probability, entropy and value estimation
        for forward pass during training
        
        Args:
            x: observations
            action: if provided, calculate the log probability of the action; otherwise sample a new action
        
        Returns:
            action: action
            log_prob: log probability of the action
            entropy: entropy of the policy (for exploration)
            value: state value estimation
        """
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    """
    supports logging to TensorBoard and Weights & Biases
    """
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    # parameter parsing and initialization
    # args = tyro.cli(Args)
    # args = Args(env_id="PushCube-v1", exp_name="rgb-pushcube-test", num_envs=256, update_epochs=8, num_minibatches=16, total_timesteps=250_000, eval_freq=10, num_steps=20)
    args = Args(env_id="MyPickSingleYCB-v1",
                exp_name=f"rgb-picksingleycb_{datetime.now().strftime('%m%d_%H%M')}",
                num_envs=1024, update_epochs=8, num_minibatches=32, total_timesteps=10_000_000, eval_freq=16, num_steps=20)

    # calculate runtime parameters
    args.batch_size = int(args.num_envs * args.num_steps)  # total number of samples per iteration
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # total number of samples per mini-batch
    args.num_iterations = args.total_timesteps // args.batch_size  # total number of iterations
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # environment setup
    env_kwargs = dict(obs_mode="rgbd", render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # use wrapper to flatten rgb observation and state keys
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # buffer for storing all data during a rollout
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # log probability of the action
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # reward
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # termination flag
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # value estimation

    # initialize environment and agent
    global_step = 0  # global step counter
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  # reset training environment
    eval_obs, _ = eval_envs.reset(seed=args.seed)  # reset evaluation environment
    next_done = torch.zeros(args.num_envs, device=device)  # environment termination flag
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)  # cumulative time statistics

    # main training loop
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)  # value estimation at the end of the episode
        agent.eval()  # set to evaluation mode
        
        # evaluation
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            # use deterministic policy for evaluation
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        # collect evaluation metrics (e.g. reward, success rate, etc.)
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # data collection (rollout)
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # algorithm logic: action selection
            with torch.no_grad():
                # use current policy to select action and estimate value
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute action and collect data
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale  # apply reward scaling

            # record training metrics (when episode ends)
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                # store observations and values at the end of the episode (for GAE calculation)
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # calculate advantage function (GAE - Generalized Advantage Estimation)
        # guide value estimation based on termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)  # value estimation at the last step
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0  # GAE cumulative term
            # calculate advantage from the last step to the first step
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # last step: use the value of the next observation
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    # other steps: use the value of the next step
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # calculate real next value: if not terminated, use nextvalues, otherwise use final_values
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to the observation at the end of the episode
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    # standard GAE calculation
                    delta = rewards[t] + args.gamma * real_next_values - values[t]  # TD 误差
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    # note: here should use next_not_terminated, but if terminated, there is no lastgaelam
            returns = advantages + values  # return = advantage + value

        # flatten batch data
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimize policy and value network
        agent.train()  # set to training mode
        b_inds = np.arange(args.batch_size)
        clipfracs = []  # record the fraction of clipped actions
        update_time = time.perf_counter()
        # update the policy and value network multiple times (PPO core idea)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # shuffle the indices
            # split the batch into multiple mini-batches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # use current policy to recalculate the log probability and value of the action
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]  # log probability ratio of the new and old policy
                ratio = logratio.exp()  # importance sampling ratio

                with torch.no_grad():
                    # calculate approximate KL divergence (for early stopping), ref: http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # if KL divergence exceeds the threshold, stop updating early
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                # normalize advantages (helps with training stability)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss (PPO clipped loss)
                pg_loss1 = -mb_advantages * ratio  # unclipped loss
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # clipped loss
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # take the larger value (conservative update)

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # value function clipped loss (optional)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # standard MSE loss
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # total loss
                entropy_loss = entropy.mean()  # entropy loss (encourages exploration)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # gradient clipping
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        
        # calculate explained variance (for evaluating the fit of the value function)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record training metrics
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        # record performance metrics
        print("SPS:", int(global_step / (time.time() - start_time)))  # Steps Per Second
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    
    # save final model
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    # clean up resources
    envs.close()
    eval_envs.close()
    if logger is not None: logger.close()
