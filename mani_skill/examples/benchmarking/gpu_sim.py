import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv
BENCHMARK_ENVS = ["FrankaPickCubeBenchmark-v1", "CartpoleBalanceBenchmark-v1", "FrankaMoveBenchmark-v1"]
@dataclass
class Args:
    """命令行参数配置类"""
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024
    cpu_sim: bool = False
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: Optional[int] = 60
    sim_freq: Optional[int] = 120
    num_cams: Optional[int] = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: Optional[int] = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: Optional[int] = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = False
    """Whether to save videos"""
    save_results: Optional[str] = None
    """Path to save results to. Should be path/to/results.csv"""

def main(args: Args):
    """
    主函数：执行 GPU 仿真基准测试
    
    主要流程：
    1. 创建环境（GPU 或 CPU 模式）
    2. 运行随机动作测试并记录性能
    3. 如果环境有预定义轨迹，运行固定轨迹测试
    4. 运行带重置的性能测试
    5. 保存视频和结果
    """
    # 初始化性能分析器
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    
    # 配置仿真参数（控制频率和仿真频率）
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    # 为基准测试环境配置相机参数
    kwargs = dict()
    if args.env_id in BENCHMARK_ENVS:
        kwargs = dict(
            camera_width=args.cam_width,
            camera_height=args.cam_height,
            num_cameras=args.num_cams,
        )
    
    # 根据配置选择 GPU 或 CPU 仿真模式
    if not args.cpu_sim:
        env = gym.make(
            args.env_id,
            num_envs=num_envs,
            obs_mode=args.obs_mode,  # state, rgb, depth or sensor_data, etc.
            render_mode=args.render_mode,
            control_mode=args.control_mode,  # pd_joint_delta_pos, pd_ee_delta_pose, pd_ee_target_delta_pose, etc.
            sim_config=sim_config,
            **kwargs
        )
        # 如果动作空间是字典类型，需要展平
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        base_env: BaseEnv = env.unwrapped
    else:
        # CPU 仿真模式：使用异步向量环境
        def make_env():
            def _init():
                env = gym.make(args.env_id,
                               obs_mode=args.obs_mode,
                               sim_config=sim_config,
                               render_mode=args.render_mode,
                               control_mode=args.control_mode,
                               **kwargs)
                env = CPUGymWrapper(env, )
                return env
            return _init
        # mac os system does not work with forkserver when using visual observations
        env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
        base_env = make_env()().unwrapped

    # 打印仿真环境详细信息
    base_env.print_sim_details()
    
    # 初始化视频保存相关变量
    images = []  # 存储渲染的图像帧
    video_nrows = int(np.sqrt(num_envs))  # 计算视频网格的行数（用于平铺多个环境的视图）
    
    # 使用推理模式（禁用梯度计算，提升性能）
    with torch.inference_mode():
        # 环境初始化和预热
        env.reset(seed=2022)
        env.step(env.action_space.sample())  # warmup step
        env.reset(seed=2022)
        
        # 如果启用视频保存，记录初始帧
        if args.save_video:
            images.append(env.render().cpu().numpy())
        
        # ========== 测试 1: 随机动作性能测试 ==========
        N = 1000  # 测试步数
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in tqdm(range(N), desc="Env Step"):
                # 生成随机动作（范围 [-1, 1]）
                actions = (
                    2 * torch.rand(env.action_space.shape, device=base_env.device)
                    - 1
                )
                # CPU 模式下需要转换为 numpy（异步向量环境处理 torch 动作很慢）
                if args.cpu_sim:
                    actions = actions.numpy() # gymnasium async vector env processes torch actions very slowly.
                
                # 执行动作并获取观察、奖励等信息
                obs, rew, terminated, truncated, info = env.step(actions)
                
                # 如果启用视频保存，记录每一帧
                if args.save_video:
                    images.append(env.render().cpu().numpy())
        
        # 记录并输出性能统计信息
        profiler.log_stats("env.step")

        # 保存随机动作测试的视频
        if args.save_video:
            # 将多个环境的图像平铺成网格
            images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
            # 保存视频到 ./videos/ms3_benchmark 目录
            images_to_video(
                images,
                output_dir="./videos/ms3_benchmark",
                video_name=f"mani_skill_gpu_sim-random_actions-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images  # 释放内存，避免 OOM

        # ========== 测试 2: 固定轨迹测试（如果环境支持）==========
        # 如果环境有预定义的固定轨迹，运行这些轨迹进行测试
        if hasattr(env.unwrapped, "fixed_trajectory"):
            for k, v in env.unwrapped.fixed_trajectory.items():
                obs, _ = env.reset()
                env.step(torch.zeros(env.action_space.shape, device=base_env.device))
                obs, _ = env.reset()
                if args.save_video:
                    images = []
                    images.append(env.render().cpu().numpy())
                actions = v["actions"]
                if v["control_mode"] == "pd_joint_pos":
                    env.unwrapped.agent.set_control_mode(v["control_mode"])
                    env.unwrapped.agent.controller.reset()
                    N = v["shake_steps"] if "shake_steps" in v else 0
                    N += sum([a[1] for a in actions])
                    with profiler.profile(f"{k}_env.step", total_steps=N, num_envs=num_envs):
                        i = 0
                        for action in actions:
                            for _ in range(action[1]):
                                a = torch.tile(action[0], (num_envs, 1))
                                if args.cpu_sim:
                                    a = a.numpy()
                                env.step(a)
                                i += 1
                                if args.save_video:
                                    images.append(env.render().cpu().numpy())
                        # runs a "shake" test, typically used to check stability of contacts/grasping
                        if "shake_steps" in v:
                            env.unwrapped.agent.set_control_mode("pd_joint_target_delta_pos")
                            env.unwrapped.agent.controller.reset()
                            while i < N:
                                actions = v["shake_action_fn"]()
                                env.step(actions)
                                if args.save_video:
                                    images.append(env.render().cpu().numpy())
                                i += 1
                    profiler.log_stats(f"{k}_env.step")
                    if args.save_video:
                        images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir="./videos/ms3_benchmark",
                            video_name=f"mani_skill_gpu_sim-fixed_trajectory={k}-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                            fps=30,
                        )
                        del images
        
        # ========== 测试 3: 带重置的性能测试 ==========
        # 测试在频繁重置环境情况下的性能
        env.reset(seed=2022)
        N = 1000
        with profiler.profile("env.step+env.reset", total_steps=N, num_envs=num_envs):
            for i in tqdm(range(N), desc="Env Step+Reset"):
                # 生成随机动作
                actions = (
                    2 * torch.rand(env.action_space.shape, device=base_env.device) - 1
                )
                if args.cpu_sim:
                    actions = actions.numpy()
                obs, rew, terminated, truncated, info = env.step(actions)
                # 每 200 步重置一次环境（测试重置性能）
                if i % 200 == 0 and i != 0:
                    env.reset()
        
        profiler.log_stats("env.step+env.reset")
        
        # 如果启用，保存示例图像（用于调试和可视化）
        if args.save_example_image:
            obs, _ = env.reset(seed=2022)
            import matplotlib.pyplot as plt
            for cam_name, cam_data in obs["sensor_data"].items():
                for k, v in cam_data.items():
                    imgs = v.cpu().numpy()
                    # 将多个环境的图像平铺
                    imgs = tile_images(imgs, nrows=int(np.sqrt(args.num_envs)))
                    cmap = None
                    # 深度图需要特殊处理
                    if k == "depth":
                        imgs[imgs == np.inf] = 0  # 将无穷大值设为 0
                        imgs = imgs[ :, :, 0]  # 取单通道
                        cmap = "gray"  # 使用灰度色彩映射
                    plt.imsave(f"maniskill_{cam_name}_{k}.png", imgs, cmap=cmap)

    # 关闭环境，释放资源
    env.close()
    
    # 如果指定了结果保存路径，将性能数据追加到 CSV 文件
    if args.save_results:
        # append results to csv
        try:
            assert (
                args.save_video == False
            ), "Saving video slows down speed a lot and it will distort results"
            
            # 创建结果目录
            Path("benchmark_results").mkdir(parents=True, exist_ok=True)
            
            # 收集测试配置数据
            data = dict(
                env_id=args.env_id,
                obs_mode=args.obs_mode,
                num_envs=args.num_envs,
                control_mode=args.control_mode,
                gpu_type=torch.cuda.get_device_name()
            )
            data.update(
                num_cameras=args.num_cams,
                camera_width=args.cam_width,
                camera_height=args.cam_height,
            )
            # 将数据追加到 CSV 文件
            profiler.update_csv(
                args.save_results,
                data,
            )
        except:
            pass

if __name__ == "__main__":
    # main(tyro.cli(Args))  # 比 argparse 更现代、更简洁的命令行参数解析, 利用 type annotations 和 dataclass 自动推导
    #  -e "PickSingleYCB-v1" -n 16  --save-video --render-mode="sensors"
    main(Args(env_id="PickSingleYCB-v1", num_envs=16, save_video=True, render_mode="sensors"))
