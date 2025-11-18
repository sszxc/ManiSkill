"""
PPO algorithm implementation for continuous action space tasks with state observations
Using ManiSkill environment for robot manipulation tasks reinforcement learning training
Reference:
https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py
"""
import warnings
warnings.filterwarnings('ignore')

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
import tyro  # command line argument parsing
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from vega_robot import Vega
from pick_single_ycb import PickSingleYCBEnv


@dataclass
class Args:
    # Experiment settings
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False  # wandb tracking
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True  # capture videos of the agent performances (check out `videos` folder)
    save_model: bool = True  # save model into the `runs/{run_name}` folder
    evaluate: bool = False  # only runs evaluation with the given model checkpoint and saves the evaluation trajectories
    checkpoint: Optional[str] = None  # path to a pretrained checkpoint

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    total_timesteps: int = 10000000
    learning_rate: float = 3e-4
    num_envs: int = 512  # the number of parallel environments
    num_eval_envs: int = 8  # the number of parallel evaluation environments
    partial_reset: bool = True  # whether to let parallel environments reset upon termination instead of truncation
    eval_partial_reset: bool = False  # whether to let parallel evaluation environments reset upon termination instead of truncation
    num_steps: int = 50  # the number of steps to run in each environment per policy rollout
    num_eval_steps: int = 50  # the number of steps to run in each evaluation environment during evaluation
    reconfiguration_freq: Optional[int] = None  # how often to reconfigure the environment during training
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"  # the control mode to use for the environment
    anneal_lr: bool = False  # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.8  # the discount factor gamma
    gae_lambda: float = 0.9  # the lambda for the general advantage estimation
    num_minibatches: int = 32  # the number of mini-batches
    num_minibatches: int = 32
    update_epochs: int = 4  # the K epochs to update the policy
    norm_adv: bool = True  # Toggles advantages normalization
    clip_coef: float = 0.2  # the surrogate clipping coefficient
    clip_vloss: bool = False  # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    ent_coef: float = 0.0  # coefficient of the entropy
    vf_coef: float = 0.5  # coefficient of the value function
    max_grad_norm: float = 0.5  # the maximum norm for the gradient clipping
    target_kl: float = 0.1  # the target KL divergence threshold for early stopping
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

class Agent(nn.Module):
    """
    PPO agent network, containing:
    - critic network: estimate state value function V(s)
    - actor network: output action mean and standard deviation (for continuous action space)
    """
    def __init__(self, envs):
        super().__init__()
        # critic network: estimate state value function V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),  # output single value scalar
        )
        # actor network: output action mean and standard deviation (for continuous action space)
        # input: observation space, output: action mean and standard deviation (for continuous action space)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),  # smaller initial standard deviation
        )
        # actor logstd: learnable parameter for action standard deviation, initialized to -0.5
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        """
        get state value function V(s)
        """
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        """
        calculate action from observations
        
        Args:
            x: observations
            deterministic: whether to use deterministic policy (return mean) or stochastic policy (sample from distribution)
        """
        action_mean = self.actor_mean(x)
        if deterministic:
            # deterministic policy: return mean (for evaluation)
            return action_mean
        # stochastic policy: sample from normal distribution
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
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    """
    supporting TensorBoard and Weights & Biases
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
    # args = Args(env_id="PushCube-v1", exp_name="state-pushcube-test", num_envs=1024, update_epochs=8, num_minibatches=32, total_timesteps=600_000, eval_freq=8, num_steps=20)

    args = Args(env_id="MyPickSingleYCB-v1",
                exp_name=f"state-picksingleycb_{datetime.now().strftime('%m%d_%H%M')}",
                num_envs=4096, update_epochs=8, num_minibatches=32, total_timesteps=10_000_000, eval_freq=16, num_steps=20)

    args.batch_size = int(args.num_envs * args.num_steps)  # total number of samples per iteration
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # total number of samples per mini-batch
    args.num_iterations = args.total_timesteps // args.batch_size  # total number of iterations
    
    # set experiment name and run name
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # environment setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    
    # flatten action space if it is a dictionary
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    
    # if video recording is enabled, add recording wrapper
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        # if training video save frequency is specified, add training video recording
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        # add evaluation video recording
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    
    # use ManiSkill vectorized environment wrapper
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    
    # ensure action space is continuous (Box type)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # get maximum episode steps
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    
    # logger initialization
    if not args.evaluate:
        print("Running training")
        # if wandb tracking is enabled
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        # initialize TensorBoard writer
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # agent and optimizer initialization
    agent = Agent(envs).to(device)  # create agent and move to specified device
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam 优化器

    # algorithm logic: storage buffer setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # observation (tuple expansion)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # action
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # log probability of the action
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # reward
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # termination flag
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # state value estimation

    # initialize environment and state
    global_step = 0  # global step counter
    start_time = time.time()  # start time
    next_obs, _ = envs.reset(seed=args.seed)  # reset training environment
    eval_obs, _ = eval_envs.reset(seed=args.seed)  # reset evaluation environment
    next_done = torch.zeros(args.num_envs, device=device)  # initialize completion flag
    
    # print configuration information
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    
    # get action space bounds, for action clipping
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        """clip action to the valid range of the action space"""
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    # if checkpoint is provided, load model weights
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # main training loop
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        
        # for storing the final state value at termination
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()  # set to evaluation mode
        
        # evaluation phase
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()  # reset evaluation environment
            eval_metrics = defaultdict(list)  # store evaluation metrics
            num_episodes = 0
            
            # run evaluation steps
            for _ in range(args.num_eval_steps):
                with torch.no_grad():  # no gradient during evaluation
                    # use deterministic policy to get action
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    # collect completed episode information
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            
            # print and record evaluation results
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")  # means running num_episodes environments in parallel
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            
            # if only evaluation mode, exit after evaluation
            if args.evaluate:
                break
        
        # save model checkpoint
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        
        # learning rate annealing
        if args.anneal_lr:
            # linear annealing: from initial learning rate to 0
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # data collection (rollout) phase
        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs  # update global step
            obs[step] = next_obs  # store current observation
            dones[step] = next_done  # store completion flag

            # algorithm logic: action selection
            with torch.no_grad():  # no gradient during data collection
                # get action, log probability and value
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()  # store state value
            actions[step] = action  # store action
            logprobs[step] = logprob  # store log probability

            # execute action and collect data
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)  # merge termination and truncation flags
            rewards[step] = reward.view(-1) * args.reward_scale  # store reward (apply scaling)

            # if episode ends, record training metrics and final state value
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                # record training metrics (e.g. episode reward, length, etc.)
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                # store state value at termination (for GAE calculation)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        # calculate advantage (GAE) and return
        # guide value estimation based on termination and truncation
        with torch.no_grad():
            # get the value of the last state (for guiding)
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)  # initialize advantage
            lastgaelam = 0  # GAE recursive term
            
            # calculate GAE (Generalized Advantage Estimation) from the last step to the first step
            for t in reversed(range(args.num_steps)):
                # determine if the next state is completed
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done  # last time step
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]  # intermediate time step
                    nextvalues = values[t + 1]
                
                # calculate real next state value
                # if the next state is not completed, use the calculated value; if completed, use the stored final value
                real_next_values = next_not_done * nextvalues + final_values[t]  # t instead of t+1
                # next_not_done is 1 means nextvalues is calculated from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is calculated based on bootstrap_at_done
                
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1:  # initialize
                        lam_coef_sum = 0.  # lambda coefficient sum
                        reward_term_sum = 0.  # the sum of the second term
                        value_term_sum = 0.  # the sum of the third term
                    
                    # if the state is completed, reset the accumulator
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    # update the accumulator
                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    # calculate advantage
                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    # standard GAE calculation
                    delta = rewards[t] + args.gamma * real_next_values - values[t]  # TD 误差
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    # note: here should use next_not_terminated, but if terminated, there is no lastgaelam
            
            # calculate return (advantage + value)
            returns = advantages + values

        # flatten batch data
        # flatten (num_steps, num_envs, ...) to (batch_size, ...)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimize policy and value network
        agent.train()  # set to training mode
        b_inds = np.arange(args.batch_size)  # batch indices
        clipfracs = []  # record the fraction of clipped actions
        update_time = time.time()
        
        # multiple updates (PPO core: update the same batch data multiple times)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # shuffle batch indices
            
            # mini-batch update
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # current mini-batch indices

                # use current policy to recalculate the log probability, entropy and value
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]  # new-old policy log probability ratio
                ratio = logratio.exp()  # probability ratio

                with torch.no_grad():
                    # calculate approximate KL divergence (for monitoring and early stopping)  ref：http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()  # old policy relative to new policy KL
                    approx_kl = ((ratio - 1) - logratio).mean()  # new policy relative to old policy KL
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  # the fraction of clipped actions

                # if KL divergence exceeds the threshold, stop updating early
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                # get the advantage of the current mini-batch
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # normalize advantage (subtract the mean, divide by the standard deviation)
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss (PPO clipped loss)
                pg_loss1 = -mb_advantages * ratio  # unclipped policy gradient loss
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # clipped policy gradient loss
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # take the larger value (conservative update)

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # clipped value loss (to prevent the value function from updating too much)
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
                # total loss = policy loss - entropy coefficient * entropy loss + value coefficient * value loss

                # backpropagation and optimization
                optimizer.zero_grad()  # zero gradients
                loss.backward()  # backpropagation
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # gradient clipping
                optimizer.step()  # update parameters

            # if KL divergence exceeds the threshold, stop updating early
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        # calculate explained variance (for monitoring the quality of the value function)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # explained variance: measure the accuracy of the value function prediction (1 means perfect prediction, 0 means meaningless prediction)

        # record training metrics
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # print and record performance metrics
        print("SPS:", int(global_step / (time.time() - start_time)))  # Steps Per Second
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    
    # clean up after training
    if not args.evaluate:
        # save final model
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    
    # close environment
    envs.close()
    eval_envs.close()
