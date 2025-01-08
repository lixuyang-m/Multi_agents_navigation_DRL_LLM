import marinenav_env.envs.marinenav_env as marinenav_env
from policy.agent import Agent
import numpy as np
import copy
import scipy.spatial
import json
from datetime import datetime
import time as t_module
import os
import matplotlib.pyplot as plt
import APF
import sys

sys.path.insert(0, "./thirdparty")
import RVO


def evaluation(state, agent, eval_env, use_rl=True, use_iqn=True, act_adaptive=True, save_episode=False):
    """评估智能体的性能
    """
    rob_num = len(eval_env.robots)  # 获取机器人数量

    # 初始化各种指标
    rewards = [0.0] * rob_num
    times = [0.0] * rob_num
    energies = [0.0] * rob_num
    collisions = [0] * rob_num  # 碰撞计数
    computation_times = []

    end_episode = False
    length = 0  # 记录当前仿真步数

    while not end_episode:
        # 从智能体中为每个机器人获取动作
        action = []
        for i, rob in enumerate(eval_env.robots):
            if rob.deactivated:  # 如果机器人已经达到目标或发生碰撞，则跳过
                action.append(None)
                continue

            assert rob.cooperative, "每个机器人必须是合作型的！"

            start = t_module.time()  # 计算动作的时间
            if use_rl:  # 使用强化学习
                if use_iqn:  # 使用IQN智能体
                    if act_adaptive:  # 是否使用自适应IQN
                        a, _, _, _ = agent.act_adaptive(state[i])
                    else:
                        a, _, _ = agent.act(state[i])
                else:  # 使用DQN智能体
                    a, _ = agent.act_dqn(state[i])
            else:  # 使用传统算法
                a = agent.act(state[i])
            end = t_module.time()
            computation_times.append(end - start)  # 记录动作计算时间

            action.append(a)

        # 在仿真环境中执行动作
        state, reward, done, info = eval_env.step(action)

        for i, rob in enumerate(eval_env.robots):
            if rob.deactivated:  # 如果机器人已停用，则跳过
                continue

            assert rob.cooperative, "每个机器人必须是合作型的！"

            # 累加奖励
            rewards[i] += agent.GAMMA ** length * reward[i]

            # 累加时间和能耗
            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])

            # 如果机器人发生碰撞，则记录碰撞次数
            if rob.collision:
                collisions[i] += 1

            # 如果机器人发生碰撞或到达目标位置，则停用
            if rob.collision or rob.reach_goal:
                rob.deactivated = True

        # 结束条件：达到最大步数或所有机器人停用
        end_episode = (length >= 360) or eval_env.check_all_deactivated()
        length += 1

    # 统计是否所有机器人都成功到达目标
    success = True if eval_env.check_all_reach_goal() else False

    # 保存成功到达目标的机器人时间和能耗数据
    success_times = []
    success_energies = []
    for i, rob in enumerate(eval_env.robots):
        if rob.reach_goal:
            success_times.append(times[i])
            success_energies.append(energies[i])

    if save_episode:  # 如果需要保存轨迹
        trajectories = []
        for rob in eval_env.robots:
            trajectories.append(copy.deepcopy(rob.trajectory))
        return success, rewards, computation_times, success_times, success_energies, trajectories, collisions
    else:
        return success, rewards, computation_times, success_times, success_energies, collisions


def exp_setup(envs, eval_schedule, i):
    """根据评估计划设置环境
    """
    observations = []

    for test_env in envs:
        # 设置环境中的合作机器人数量、非合作机器人数量、障碍物数量和起点到终点的最小距离
        test_env.num_cooperative = eval_schedule["num_cooperative"][i]
        test_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
        test_env.num_obs = eval_schedule["num_obstacles"][i]
        test_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

        # 重置环境并保存初始状态
        state, _, _ = test_env.reset()
        observations.append(state)

    return observations


def dashboard(eval_schedule, i):
    """打印评估计划
    """
    print("\n======== 评估计划 ========")
    print("合作型智能体数量: ", eval_schedule["num_cooperative"][i])
    print("非合作型智能体数量: ", eval_schedule["num_non_cooperative"][i])
    print("障碍物数量: ", eval_schedule["num_obstacles"][i])
    print("起点到终点的最小距离: ", eval_schedule["min_start_goal_dis"][i])
    print("======== 评估计划 ========\n")


def run_experiment(eval_schedules):
    """运行实验
    """
    # 定义智能体和环境
    # agents = [adaptive_IQN_agent,IQN_agent,DQN_agent,APF_agent,RVO_agent]
    # names = ["adaptive_IQN","IQN","DQN","APF","RVO"]
    agents = [IQN_agent,DQN_agent,APF_agent,RVO_agent]
    names = ["IQN","DQN","APF","RVO"]
    # agents = [adaptive_IQN_agent]
    # names = ["adaptive_IQN"]
    envs = [test_env_0, test_env_1, test_env_2, test_env_3, test_env_4]
    evaluations = [evaluation, evaluation, evaluation, evaluation, evaluation]

    colors = ["b", "g", "r", "tab:orange", "m"]  # 图表中使用的颜色

    save_trajectory = True  # 是否保存轨迹

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")  # 获取当前时间戳

    # 初始化实验数据
    robot_nums = []
    all_successes_exp = []
    all_rewards_exp = []
    all_success_times_exp = []
    all_success_energies_exp = []
    all_collision_counts_exp = []
    if save_trajectory:
        all_trajectories_exp = []
        all_eval_configs_exp = []

    # 遍历每个评估计划
    for idx, count in enumerate(eval_schedules["num_episodes"]):
        dashboard(eval_schedules, idx)

        robot_nums.append(eval_schedules["num_cooperative"][idx])  # 记录合作型机器人数量
        all_successes = [[] for _ in agents]
        all_rewards = [[] for _ in agents]
        all_computation_times = [[] for _ in agents]
        all_success_times = [[] for _ in agents]
        all_success_energies = [[] for _ in agents]
        all_collision_counts = [[] for _ in agents]
        if save_trajectory:
            all_trajectories = [[] for _ in agents]
            all_eval_configs = [[] for _ in agents]

        # 对每个智能体运行指定次数的评估
        for i in range(count):
            print("正在评估所有智能体，第 ", i, " 次")
            observations = exp_setup(envs, eval_schedules, idx)
            for j in range(len(agents)):
                agent = agents[j]
                env = envs[j]
                eval_func = evaluations[j]
                name = names[j]

                if save_trajectory:
                    all_eval_configs[j].append(env.episode_data())

                # 获取初始状态
                obs = observations[j]
                if save_trajectory:
                    if name == "adaptive_IQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories, collisions = eval_func(
                            obs, agent, env, save_episode=True)
                    elif name == "IQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories, collisions = eval_func(
                            obs, agent, env, act_adaptive=False, save_episode=True)
                    elif name == "DQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories, collisions = eval_func(
                            obs, agent, env, use_iqn=False, save_episode=True)
                    elif name == "APF":
                        success, rewards, computation_times, success_times, success_energies, trajectories, collisions = eval_func(
                            obs, agent, env, use_rl=False, save_episode=True)
                    elif name == "RVO":
                        success, rewards, computation_times, success_times, success_energies, trajectories, collisions = eval_func(
                            obs, agent, env, use_rl=False, save_episode=True)
                    else:
                        raise RuntimeError("未实现的智能体！")
                else:
                    if name == "adaptive_IQN":
                        success, rewards, computation_times, success_times, success_energies, collisions = eval_func(obs, agent,
                                                                                                         env)
                    elif name == "IQN":
                        success, rewards, computation_times, success_times, success_energies, collisions = eval_func(obs, agent,
                                                                                                         env,
                                                                                                         act_adaptive=False)
                    elif name == "DQN":
                        success, rewards, computation_times, success_times, success_energies, collisions = eval_func(obs, agent,
                                                                                                         env,
                                                                                                         use_iqn=False)
                    elif name == "APF":
                        success, rewards, computation_times, success_times, success_energies, collisions = eval_func(obs, agent,
                                                                                                         env,
                                                                                                         use_rl=False)
                    elif name == "RVO":
                        success, rewards, computation_times, success_times, success_energies, collisions = eval_func(obs, agent,
                                                                                                         env,
                                                                                                         use_rl=False)
                    else:
                        raise RuntimeError("未实现的智能体！")

                all_successes[j].append(success)
                all_rewards[j] += rewards
                all_computation_times[j] += computation_times
                all_success_times[j] += success_times
                all_success_energies[j] += success_energies
                all_collision_counts[j] += collisions
                if save_trajectory:
                    all_trajectories[j].append(copy.deepcopy(trajectories))

        # 计算和打印每个智能体的评估结果
        for k, name in enumerate(names):
            s_rate = np.sum(all_successes[k]) / len(all_successes[k])  # 成功率
            avg_r = np.mean(all_rewards[k])  # 平均奖励
            avg_compute_t = np.mean(all_computation_times[k])  # 平均计算时间
            avg_t = np.mean(all_success_times[k])  # 平均成功时间
            avg_e = np.mean(all_success_energies[k])  # 平均能耗
            total_collisions = np.sum(all_collision_counts[k])  # 总碰撞次数
            total_robots = count * eval_schedules["num_cooperative"][idx]  # 总的智能体数量
            collision_rate = total_collisions / total_robots  # 碰撞率
            print(f"{name} | 成功率: {s_rate:.2f} | 碰撞率: {collision_rate:.2f} | 平均奖励: {avg_r:.2f} | 平均计算时间: {avg_compute_t} | \
                  平均时间: {avg_t:.2f} | 平均能耗: {avg_e:.2f}")

        print("\n")

        all_successes_exp.append(all_successes)
        all_rewards_exp.append(all_rewards)
        all_success_times_exp.append(all_success_times)
        all_success_energies_exp.append(all_success_energies)
        if save_trajectory:
            all_trajectories_exp.append(copy.deepcopy(all_trajectories))
            all_eval_configs_exp.append(copy.deepcopy(all_eval_configs))

    # 保存实验数据
    if save_trajectory:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                        all_trajectories_exp=all_trajectories_exp,
                        all_eval_configs_exp=all_eval_configs_exp
                        )
    else:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                        )

    # 创建实验结果保存目录
    exp_dir = f"experiment_data/exp_data_{timestamp}"
    os.makedirs(exp_dir)

    filename = os.path.join(exp_dir, "exp_results.json")
    with open(filename, "w") as file:
        json.dump(exp_data, file)

    # 可视化结果
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    bar_width = 0.25
    interval_scale = 1.5
    set_label = [True] * len(names)
    for i, robot_num in enumerate(robot_nums):

        all_successes = all_successes_exp[i]
        all_success_times = all_success_times_exp[i]
        all_success_energies = all_success_energies_exp[i]
        for j, pos in enumerate([-2 * bar_width, -bar_width, 0.0, bar_width, 2 * bar_width]):
            # 成功率柱状图
            s_rate = np.sum(all_successes[j]) / len(all_successes[j])
            if set_label[j]:
                ax1.bar(interval_scale * i + pos, s_rate, 0.8 * bar_width, color=colors[j], label=names[j])
                set_label[j] = False
            else:
                ax1.bar(interval_scale * i + pos, s_rate, 0.8 * bar_width, color=colors[j])

            # 时间箱线图
            box = ax2.boxplot(all_success_times[j], positions=[interval_scale * i + pos],
                              flierprops={'marker': '.', 'markersize': 1}, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

            # 能耗箱线图
            box = ax3.boxplot(all_success_energies[j], positions=[interval_scale * i + pos],
                              flierprops={'marker': '.', 'markersize': 1}, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

    # 设置图表标签和标题
    ax1.set_xticks(interval_scale * np.arange(len(robot_nums)))
    ax1.set_xticklabels(robot_nums)
    ax1.set_title("成功率")
    ax1.legend()

    ax2.set_xticks(interval_scale * np.arange(len(robot_nums)))
    ax2.set_xticklabels([str(robot_num) for robot_num in eval_schedules["num_cooperative"]])
    ax2.set_title("时间")

    ax3.set_xticks(interval_scale * np.arange(len(robot_nums)))
    ax3.set_xticklabels([str(robot_num) for robot_num in eval_schedules["num_cooperative"]])
    ax3.set_title("能耗")

    # 保存图表
    fig1.savefig(os.path.join(exp_dir, "success_rate.png"))
    fig2.savefig(os.path.join(exp_dir, "time.png"))
    fig3.savefig(os.path.join(exp_dir, "energy.png"))


if __name__ == "__main__":
    seed = 3  # 用于测试环境的随机数种子

    '''
    自适应 IQN 智能体
    '''
    test_env_0 = marinenav_env.MarineNavEnv2(seed)
    save_dir = "pretrained_models/IQN/seed_9"
    device = "cpu"
    adaptive_IQN_agent = Agent(cooperative=True, device=device)
    adaptive_IQN_agent.load_model(save_dir, "cooperative", device)

    '''
    IQN 智能体
    '''
    test_env_1 = marinenav_env.MarineNavEnv2(seed)
    save_dir = "pretrained_models/IQN/seed_9"
    device = "cpu"
    IQN_agent = Agent(cooperative=True, device=device)
    IQN_agent.load_model(save_dir, "cooperative", device)

    '''
    DQN 智能体
    '''
    test_env_2 = marinenav_env.MarineNavEnv2(seed)
    save_dir = "pretrained_models/DQN/seed_9"
    device = "cpu"
    DQN_agent = Agent(cooperative=True, device=device, use_iqn=False)
    DQN_agent.load_model(save_dir, "cooperative", device)

    '''
    APF 智能体
    '''
    test_env_3 = marinenav_env.MarineNavEnv2(seed)
    APF_agent = APF.APF_agent(test_env_3.robots[0].a, test_env_3.robots[0].w)

    '''
    RVO 智能体
    '''
    test_env_4 = marinenav_env.MarineNavEnv2(seed)
    RVO_agent = RVO.RVO_agent(test_env_4.robots[0].a, test_env_4.robots[0].w, test_env_4.robots[0].max_speed)

    # 定义评估计划
    eval_schedules = dict(num_episodes=[100, 100, 100, 100, 100],  # 每个环境的评估次数
                          # num_cooperative=[3, 4, 5, 6, 7],  # 合作型智能体数量
                          num_cooperative=[8, 9, 10, 11, 12],  # 合作型智能体数量
                          num_non_cooperative=[0, 0, 0, 0, 0],  # 非合作型智能体数量
                          num_obstacles=[8, 8, 8, 8, 8],  # 障碍物数量
                          min_start_goal_dis=[40.0, 40.0, 40.0, 40.0, 40.0]  # 起点到终点的最小距离
                          )

    run_experiment(eval_schedules)