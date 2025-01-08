import json
import numpy as np
import os
import copy
import time
import random
import torch

# 设置自定义缓存目录
os.environ["HF_HOME"] = "/data2/lxy"
# 只让 GPU 7 可见
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


class Trainer():
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,  # 评估设置
                 non_cooperative_agent=None,
                 cooperative_agent=None,
                 UPDATE_EVERY=4,  # 每隔多少步执行一次训练
                 learning_starts=2000,  # 在多少步后开始训练
                 target_update_interval=10000,  # 每隔多少步更新目标网络
                 exploration_fraction=0.25,  # 探索阶段所占的总步数比例
                 initial_eps=0.6,  # 初始探索率
                 final_eps=0.05,  # 最终探索率
                 seed=0
                 ):
        self.seed = seed
        self.set_seed(seed)

        # 初始化环境和智能体
        self.train_env = train_env
        self.eval_env = eval_env
        self.cooperative_agent = cooperative_agent
        self.noncooperative_agent = non_cooperative_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        # 评估配置
        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        # 初始化时间步长
        self.current_timestep = 0

        # 已学习的时间步（从 `learning_starts` 开始计数）
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_actions = []
        self.eval_trajectories = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_obs = []
        self.eval_objs = []

        self.vertical_actions = ["decelerate", "keep", "accelerate"]
        self.horizontal_actions = ["turn left", "keep", "turn right"]
        # 定义动作强度的权重映射
        self.intensity_weights = {
            "slightly": 0.6,  # 稍微的动作分配较小的权重
            "normal": 1.0,  # 普通动作分配标准权重
            "strongly": 1.5  # 大幅动作分配较大的权重
        }
        self.action_space = [(v, h) for v in self.vertical_actions for h in self.horizontal_actions]
    
    def set_seed(self, seed):
        """
        设置随机种子以保证结果的可复现性
        """
        # 固定 Python 原生随机数生成器
        random.seed(seed)

        # 固定 NumPy 的随机数生成器
        np.random.seed(seed)

        # 固定 PyTorch 的随机数生成器（CPU 和 GPU）
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def check_collision_in_future(self, state, ttc=0.2, robot_radius=0.8):
        """
        判断在未来指定时间点 (0.25s) 时机器人是否会与静态或动态障碍物发生碰撞。

        Parameters:
        - state (list): 输入的机器人坐标系下的状态，包含机器人自身状态、静态障碍物和动态障碍物。
            格式为: [goal_x, goal_y, velocity_x, velocity_y,
                     static_obs_1.x, static_obs_1.y, static_obs_1.r, ..., static_obs_5.x, static_obs_5.y, static_obs_5.r,
                     dynamic_obs_1.x, dynamic_obs_1.y, dynamic_obs_1.vx, dynamic_obs_1.vy, ..., dynamic_obs_5.x, dynamic_obs_5.y, dynamic_obs_5.vx, dynamic_obs_5.vy]
            缺失或多余部分用 0 填充。
        - ttc (float): 时间预测范围（秒），默认为 0.25s。

        Returns:
        - bool: 是否发生碰撞，True 表示会发生碰撞，False 表示不会发生碰撞。
        """
        # 提取机器人自身状态
        ego_state = state[:4]
        goal_x, goal_y, velocity_x, velocity_y = ego_state

        # 提取静态障碍物状态
        static_obstacles = np.array(state[4:4 + 5 * 3]).reshape(5, 3)

        # 提取动态障碍物状态
        dynamic_obstacles = np.array(state[4 + 5 * 3:]).reshape(5, 4)  # 每个动态障碍物 [x, y, vx, vy]

        robot_position = np.array([0.0, 0.0])
        robot_velocity = np.array([velocity_x, velocity_y])
        robot_future_position = robot_position + robot_velocity * ttc

        for obs in static_obstacles:
            obs_x, obs_y, obs_r = obs
            if obs_x == 0.0 and obs_y == 0.0 and obs_r == 0.0:
                continue  # 跳过无效障碍物

            obs_position = np.array([obs_x, obs_y])
            distance = np.linalg.norm(robot_future_position - obs_position) - (obs_r + robot_radius)

            if distance <= 0:
                return True

        for obs in dynamic_obstacles:
            obs_x, obs_y, obs_vx, obs_vy = obs
            if obs_x == 0.0 and obs_y == 0.0 and obs_vx == 0.0 and obs_vy == 0.0:
                continue  # 跳过无效障碍物

            obs_velocity = np.array([obs_vx, obs_vy])
            obs_position = np.array([obs_x, obs_y])
            obs_future_position = obs_position + obs_velocity * ttc
            distance = np.linalg.norm(robot_future_position - obs_future_position) - (robot_radius * 2)

            if distance <= 0:
                return True

        return False  # 没有碰撞发生

    def create_eval_configs(self, eval_schedule):
        self.eval_config.clear()

        count = 0
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                self.eval_env.num_cooperative = eval_schedule["num_cooperative"][i]
                self.eval_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

                self.eval_env.reset()

                # save eval config
                self.eval_config.append(self.eval_env.episode_data())
                count += 1

    def save_eval_config(self, directory):
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def generate_prompt_local(self, states, coll_robots_list):
        num_robot = len(coll_robots_list)

        # 所有车辆的信息
        all_vehicles = []
        all_obstacles = []
        all_goal_x = []
        all_goal_y = []
        all_vx = []
        all_vy = []

        # 遍历所有车辆
        for i in range(num_robot):
            # 自车信息：
            goal_x = states[coll_robots_list[i]][0]
            goal_y = states[coll_robots_list[i]][1]
            vx = states[coll_robots_list[i]][2]
            vy = states[coll_robots_list[i]][3]
            all_goal_x.append(goal_x)
            all_goal_y.append(goal_y)
            all_vx.append(vx)
            all_vy.append(vy)

            # 其他车辆的信息
            vehicles = []
            for j in range(19, 39, 4):
                x = states[coll_robots_list[i]][j]
                y = states[coll_robots_list[i]][j + 1]
                vx = states[coll_robots_list[i]][j + 2]
                vy = states[coll_robots_list[i]][j + 3]
                if x == 0 and y == 0 and vx == 0 and vy == 0:
                    continue  # 跳过填充的0
                veh_number = (j - 19) // 4 + 1
                vehicles.append(
                    f"Target {veh_number}: relative position ({x:.1f}, {y:.1f}), relative velocity ({vx:.1f}, {vy:.1f})")
            all_vehicles.append(vehicles)

            # Obstacles
            obstacles = []
            for j in range(4, 19, 3):
                ox = states[coll_robots_list[i]][j]
                oy = states[coll_robots_list[i]][j + 1]
                radius = states[coll_robots_list[i]][j + 2]
                if ox == 0 and oy == 0 and radius == 0:
                    continue  # 跳过填充的0
                obs_number = (j - 4) // 3 + 1
                obstacles.append(
                    f"Obstacle {obs_number}: center relative position ({ox:.1f}, {oy:.1f}), radius {radius}")
            all_obstacles.append(obstacles)

        # Construct the prompt
        prompt = (
            "You are an expert in multi-agent path planning, providing high-level planning suggestions for multiple RL-based vehicles. "
            "Your tasks include lateral control (turn left/keep/turn right) and longitudinal control (accelerate/decelerate/keep).\n"
            "The objectives of high-level planning are: avoid collisions, maintain safe distances, and drive as quickly as possible while ensuring safety (aim to accelerate to 2 m/s if safe).\n"
            "For turning and accelerating/decelerating, you can use [slightly] and [strongly] to indicate the degree of action.\n"
            "All turning and accelerating/deaccelerating suggestions are based on each vehicle's local coordinate system."
            "The forward direction of each vehicle is considered the positive X-axis, and the left-hand side is the positive Y-axis.\n"
            "Output format: Each vehicle's suggestion must be on a new line:\n"
            "Vehicle 1: [slightly turn left/turn left/strongly turn left/keep/slightly turn right/turn right/strongly turn right],"
            "[slightly accelerate/accelerate/strongly accelerate/keep/slightly decelerate/decelerate/strongly decelerate].\n"
            "Vehicle 2: [slightly turn left/turn left/strongly turn left/keep/slightly turn right/turn right/strongly turn right],"
            "[slightly accelerate/accelerate/strongly accelerate/keep/slightly decelerate/decelerate/strongly decelerate].\n"
            "...\n"
            "Please provide suggestions directly without repeating vehicle positions and velocities. Avoid additional explanations.\n\n"
        )

        # Add information for each vehicle in a structured format
        for i in range(len(coll_robots_list)):
            if states[i] is None:
                continue
            prompt += f"========== Vehicle {coll_robots_list[i] + 1} ==========\n"
            prompt += f"Goal relative position: ({all_goal_x[i]:.1f}, {all_goal_y[i]:.1f})\n"
            prompt += f"Goal relative velocity: ({all_vx[i]:.1f}, {all_vy[i]:.1f})\n"

            prompt += "Dynamic targets observed by Vehicle:\n"
            if len(all_vehicles[i]) == 0:
                prompt += "    No dynamic targets within perception range.\n"
            else:
                for veh in all_vehicles[i]:
                    prompt += f"    {veh}\n"

            prompt += "Static obstacles observed by Vehicle:\n"
            if len(all_obstacles[i]) == 0:
                prompt += "    No obstacles within perception range.\n"
            else:
                for obs in all_obstacles[i]:
                    prompt += f"    {obs}\n"

        return prompt

    def parse_model_suggestion(self, suggestion):
        """
        Parse the LLM suggestion and extract lateral and longitudinal actions with their intensities.
        """
        horizontal_strength = None
        vertical_strength = None

        if "slightly turn left" in suggestion:
            horizontal_action = "turn left"
            horizontal_strength = "slightly"
        elif "turn left" in suggestion:
            horizontal_action = "turn left"
            horizontal_strength = "normal"
        elif "strongly turn left" in suggestion:
            horizontal_action = "turn left"
            horizontal_strength = "strongly"
        elif "slightly turn right" in suggestion:
            horizontal_action = "turn right"
            horizontal_strength = "slightly"
        elif "turn right" in suggestion:
            horizontal_action = "turn right"
            horizontal_strength = "normal"
        elif "strongly turn right" in suggestion:
            horizontal_action = "turn right"
            horizontal_strength = "strongly"
        else:
            horizontal_action = "keep"
            horizontal_strength = "normal"

        if "slightly accelerate" in suggestion:
            vertical_action = "accelerate"
            vertical_strength = "slightly"
        elif "accelerate" in suggestion:
            vertical_action = "accelerate"
            vertical_strength = "normal"
        elif "strongly accelerate" in suggestion:
            vertical_action = "accelerate"
            vertical_strength = "strongly"
        elif "slightly decelerate" in suggestion:
            vertical_action = "decelerate"
            vertical_strength = "slightly"
        elif "decelerate" in suggestion:
            vertical_action = "decelerate"
            vertical_strength = "normal"
        elif "strongly decelerate" in suggestion:
            vertical_action = "decelerate"
            vertical_strength = "strongly"
        else:
            vertical_action = "keep"
            vertical_strength = "normal"

        return (horizontal_action, horizontal_strength, vertical_action, vertical_strength)

    def generate_action_distribution(self, suggestion):
        """
        Generate a 9-dimensional action space probability distribution based on LLM suggestions.
        """
        # Parse the suggestion
        horizontal_action, horizontal_strength, vertical_action, vertical_strength = self.parse_model_suggestion(
            suggestion)

        # Initialize probability distribution
        probabilities = np.zeros(len(self.action_space))

        # Get intensity weights
        h_weight = self.intensity_weights[horizontal_strength]
        v_weight = self.intensity_weights[vertical_strength]

        # Assign probabilities
        for i, (v, h) in enumerate(self.action_space):
            if h == horizontal_action and v == vertical_action:
                probabilities[i] = h_weight * v_weight
            elif h == horizontal_action:
                probabilities[i] = h_weight * 0.5  # Partial match for horizontal action
            elif v == vertical_action:
                probabilities[i] = v_weight * 0.5  # Partial match for vertical action
            else:
                probabilities[i] = 0.1  # Minimal probability for other actions

        # Normalize
        probabilities /= probabilities.sum()
        return probabilities

    def learn(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):

        states, _, _ = self.train_env.reset()
        num_robots = len(states)

        # current episode
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_deactivated_t = [-1] * len(self.train_env.robots)
        ep_length = 0
        ep_num = 0

        llm_guidance_timestep_count = 0  # 用于记录 LLM guidance 持续的步数

        # 配置 TurboMind 后端和生成参数
        backend_config = TurbomindEngineConfig(cache_max_entry_count=0.5, tp=1)  # 单 GPU 模式
        gen_config = GenerationConfig(
            top_p=1.0,
            temperature=0.0,
            max_new_tokens=1024,
            random_seed=self.seed
        )
        # 加载 pipeline
        pipe = pipeline('internlm/internlm2_5-7b-chat', backend_config=backend_config)
        # pipe = pipeline('Qwen/Qwen2.5-7B-Instruct', backend_config=backend_config)

        llm_guidance_times = 0
        pri_sample_flag = False  # 是否对llm指导的经验进行优先经验回放

        while self.current_timestep <= total_timesteps:
            coll_robots_list = []
            # 计算每个robot的TTC
            for i in range(num_robots):
                if states[i] is not None:
                    if_coll = self.check_collision_in_future(states[i])
                    if if_coll:  # 如果第i个robot在未来发生了碰撞
                        coll_robots_list.append(i)

            eps = self.linear_eps(total_timesteps)

            actions = []
            if self.cooperative_agent.use_llm:
                if len(coll_robots_list) != 0 and llm_guidance_timestep_count == 0:
                    # 获取车辆的局部坐标系信息
                    prompt = self.generate_prompt_local(states, coll_robots_list)
                    # print(prompt)

                    prompts = [[{
                        'role': 'user',
                        'content': prompt
                    }]]

                    # 调用 pipeline 推理并打印结果
                    # response = pipe(prompts, gen_config=gen_config)
                    # response_lines = response[0].text.splitlines()  # 按行分割文本


                    # print(response_lines)
                    llm_probabilities = []
                    j = 0
                    # print("coll_robots_list:", coll_robots_list)
                    # print(len(response_lines), response_lines)
                    try:
                        for i in range(len(states)):
                            # if i in coll_robots_list:
                            #     # print("not None", i)
                            #     llm_probabilitie = self.generate_action_distribution(response_lines[j])
                            #     j += 1
                            # else:
                            #     # print("None", i)
                            #     llm_probabilitie = None


                            llm_probabilitie = None

                            # print("llm_probabilitie:", llm_probabilitie)
                            llm_probabilities.append(llm_probabilitie)
                            # print(llm_probabilities)
                        llm_guidance_timestep_count = 5  # 用于记录 LLM guidance 持续的步数
                        llm_guidance_times += 1
                        # print("大模型指导次数:", llm_guidance_times)
                    except Exception as e:
                        print(f"Error occurred at index: {e}")
                        for i in range(len(states)):
                            if states[i] is None:
                                print(states[i], "is None")
                        print(len(states), coll_robots_list, "prompt:", prompts, "response:", response_lines, "length:", len(response_lines), "llm_probabilities:", llm_probabilities)
                        llm_guidance_timestep_count = 0  # 用于记录 LLM guidance 持续的步数

            for i, rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    actions.append(None)
                    continue

                if rob.cooperative:
                    if self.cooperative_agent.use_iqn:
                        # 如果混合大模型指导
                        if self.cooperative_agent.use_llm:
                            # 间歇性引入大模型指导
                            if llm_guidance_timestep_count > 0:
                                action, _, _, rl_values, llm_values, llm_weight = self.cooperative_agent.act_llm(
                                    states[i], eps, llm_action=llm_probabilities[i])
                            else:
                                action, _, _, rl_values, llm_values, llm_weight = self.cooperative_agent.act_llm(
                                    states[i], eps, llm_action=None)
                            
                            # action, _, _, rl_values, llm_values, llm_weight = self.cooperative_agent.act_llm(
                            #         states[i], eps, llm_action=None)
                        else:
                            action, _, _ = self.cooperative_agent.act(states[i], eps)
                    else:
                        action, _ = self.cooperative_agent.act_dqn(states[i], eps)
                else:
                    if self.noncooperative_agent.use_iqn:
                        action, _, _ = self.noncooperative_agent.act(states[i], eps)
                    else:
                        action, _ = self.noncooperative_agent.act_dqn(states[i], eps)
                actions.append(action)

            next_states, rewards, dones, infos = self.train_env.step(actions, coll_robots_list)

            # save experience in replay memory
            if pri_sample_flag:
                for i, rob in enumerate(self.train_env.robots):
                    if rob.deactivated:
                        continue

                    # 判断经验类型（普通 or LLM 指导）
                    if rob.cooperative:
                        ep_rewards[i] += self.cooperative_agent.GAMMA ** ep_length * rewards[i]

                        # 判断是否使用 LLM 指导
                        if llm_guidance_timestep_count > 0 and llm_probabilities[i] is not None:
                            # 存储到 LLM 指导经验池
                            self.cooperative_agent.dual_memory.add(
                                (states[i], actions[i], rewards[i], next_states[i], dones[i]), llm_guided=True)
                        else:
                            # 存储到普通经验池
                            self.cooperative_agent.dual_memory.add(
                                (states[i], actions[i], rewards[i], next_states[i], dones[i]), llm_guided=False)
                    else:
                        ep_rewards[i] += self.noncooperative_agent.GAMMA ** ep_length * rewards[i]
                        # 非合作智能体只存储到普通经验池
                        self.noncooperative_agent.normal_memory.add(
                            (states[i], actions[i], rewards[i], next_states[i], dones[i]))
                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True
                        ep_deactivated_t[i] = ep_length
            else:
                for i, rob in enumerate(self.train_env.robots):
                    if rob.deactivated:
                        continue

                    if rob.cooperative:
                        ep_rewards[i] += self.cooperative_agent.GAMMA ** ep_length * rewards[i]
                        if self.cooperative_agent.training:
                            self.cooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                    else:
                        ep_rewards[i] += self.noncooperative_agent.GAMMA ** ep_length * rewards[i]
                        if self.noncooperative_agent.training:
                            self.noncooperative_agent.memory.add(
                                (states[i], actions[i], rewards[i], next_states[i], dones[i]))

                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True
                        ep_deactivated_t[i] = ep_length

            end_episode = (ep_length >= 1000) or self.train_env.check_all_deactivated()

            # Learn, update and evaluate models after learning_starts time step
            if self.current_timestep >= self.learning_starts:
                # start_3 = time.time()

                for agent in [self.cooperative_agent, self.noncooperative_agent]:
                    if agent is None:
                        continue

                    if not agent.training:
                        continue

                    # Learn every UPDATE_EVERY time steps.
                    if self.current_timestep % self.UPDATE_EVERY == 0:
                        # If enough samples are available in memory, get random subset and learn
                        if agent.memory.size() > agent.BATCH_SIZE:
                            if self.cooperative_agent.use_llm:
                                agent.train(rl_values, llm_values, llm_weight)
                            else:
                                agent.train(rl_values, None, None)

                    # Update the target model every target_update_interval time steps
                    if self.current_timestep % self.target_update_interval == 0:
                        agent.soft_update()

                if self.current_timestep % eval_freq == 0:
                    self.evaluation()
                    self.save_evaluation(eval_log_path)

                    for agent in [self.cooperative_agent, self.noncooperative_agent]:
                        if agent is None:
                            continue
                        if not agent.training:
                            continue
                        # save the latest models
                        agent.save_latest_model(eval_log_path)

            if llm_guidance_timestep_count > 0:
                llm_guidance_timestep_count -= 1

            if end_episode:
                ep_num += 1

                if verbose:
                    # print abstract info of learning process
                    print("======== Episode Info ========")
                    print("current ep_length: ", ep_length)
                    print("current ep_num: ", ep_num)
                    print("current exploration rate: ", eps)
                    print("current timesteps: ", self.current_timestep)
                    print("total timesteps: ", total_timesteps)
                    print("llm_guidance_times:", llm_guidance_times)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for i, rob in enumerate(self.train_env.robots):
                        info = infos[i]["state"]
                        if info == "deactivated after collision" or info == "deactivated after reaching goal":
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
                        else:
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                    print("======== Robots Info ========\n")

                states, _, _ = self.train_env.reset()
                llm_guidance_timestep_count = 0  # 用于记录 LLM guidance 持续的步数

                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_deactivated_t = [-1] * len(self.train_env.robots)
                ep_length = 0
            else:
                states = next_states
                ep_length += 1

            self.current_timestep += 1

    def linear_eps(self, total_timesteps):

        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def evaluation(self):
        """Evaluate performance of the agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            eval_config: eval envs config file
        """
        actions_data = []
        trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        obs_data = []
        objs_data = []

        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            state, _, _ = self.eval_env.reset_with_eval_config(config)
            obs = [[copy.deepcopy(rob.perception.observed_obs)] for rob in self.eval_env.robots]
            objs = [[copy.deepcopy(rob.perception.observed_objs)] for rob in self.eval_env.robots]

            rob_num = len(self.eval_env.robots)

            rewards = [0.0] * rob_num
            times = [0.0] * rob_num
            energies = [0.0] * rob_num
            end_episode = False
            length = 0

            while not end_episode:
                # gather actions for robots from agents
                action = []
                for i, rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        action.append(None)
                        continue

                    if rob.cooperative:
                        if self.cooperative_agent.use_iqn:
                            a, _, _ = self.cooperative_agent.act(state[i])
                        else:
                            a, _ = self.cooperative_agent.act_dqn(state[i])
                    else:
                        if self.noncooperative_agent.use_iqn:
                            a, _, _ = self.noncooperative_agent.act(state[i])
                        else:
                            a, _ = self.noncooperative_agent.act_dqn(state[i])

                    action.append(a)

                # execute actions in the training environment
                state, reward, done, info = self.eval_env.step(action, coll_robots_list=[])

                for i, rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        continue

                    if rob.cooperative:
                        rewards[i] += self.cooperative_agent.GAMMA ** length * reward[i]
                    else:
                        rewards[i] += self.noncooperative_agent.GAMMA ** length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_action_energy_cost(action[i])
                    obs[i].append(copy.deepcopy(rob.perception.observed_obs))
                    objs[i].append(copy.deepcopy(rob.perception.observed_objs))

                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True

                end_episode = (length >= 1000) or self.eval_env.check_any_collision() or self.eval_env.check_all_deactivated()
                length += 1

            actions = []
            trajectories = []
            for rob in self.eval_env.robots:
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))

            success = True if self.eval_env.check_all_reach_goal() else False

            actions_data.append(actions)
            trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            obs_data.append(obs)
            objs_data.append(objs)

        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data) / len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])

        print(f"++++++++ Evaluation Info ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        if avg_t is not None:
            print(f"Avg time: {avg_t:.2f}")
            print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        self.eval_actions.append(actions_data)
        self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        self.eval_obs.append(obs_data)
        self.eval_objs.append(objs_data)

    def save_evaluation(self, eval_log_path):
        filename = "evaluations.npz"

        np.savez(
            os.path.join(eval_log_path, filename),
            timesteps=np.array(self.eval_timesteps, dtype=object),
            actions=np.array(self.eval_actions, dtype=object),
            trajectories=np.array(self.eval_trajectories, dtype=object),
            rewards=np.array(self.eval_rewards, dtype=object),
            successes=np.array(self.eval_successes, dtype=object),
            times=np.array(self.eval_times, dtype=object),
            energies=np.array(self.eval_energies, dtype=object),
            obs=np.array(self.eval_obs, dtype=object),
            objs=np.array(self.eval_objs, dtype=object)
        )