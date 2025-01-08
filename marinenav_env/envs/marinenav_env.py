import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
import gym
import json
import copy

class Obstacle:

    def __init__(self, x: float, y: float, r: float):
        self.x = x  # 障碍物中心的x坐标
        self.y = y  # 障碍物中心的y坐标
        self.r = r  # 障碍物的半径

class MarineNavEnv2(gym.Env):

    def __init__(self, seed: int = 0, schedule: dict = None):
        self.sd = seed
        self.rd = np.random.RandomState(seed)  # 伪随机数生成器

        # 参数初始化
        self.width = 50  # 地图的x轴维度
        self.height = 50  # 地图的y轴维度
        self.obs_r_range = [1, 1]  # 障碍物半径的范围
        self.clear_r = 5.0  # 机器人起点（终点）周围无障碍物的半径范围
        self.timestep_penalty = -1.0  # 时间步惩罚
        self.collision_penalty = -50.0  # 碰撞惩罚
        self.goal_reward = 100.0  # 到达目标的奖励
        self.num_obs = 8  # 静态障碍物的数量
        self.min_start_goal_dis = 30.0  # 起点和终点的最小距离
        self.num_cooperative = 3  # 协作机器人的数量
        self.num_non_cooperative = 3  # 非协作机器人的数量

        self.ttc_penalty = -1.0

        # 初始化机器人列表
        self.robots = []
        for _ in range(self.num_cooperative):
            self.robots.append(robot.Robot(cooperative=True))
        for _ in range(self.num_non_cooperative):
            self.robots.append(robot.Robot(cooperative=False))

        # 初始化障碍物列表
        self.obstacles = []

        self.schedule = schedule  # 课程学习的计划
        self.episode_timesteps = 0  # 当前episode的时间步数
        self.total_timesteps = 0  # 总的学习时间步数

        self.observation_in_robot_frame = True  # 返回机器人坐标系下的观测值

    def get_action_space_dimension(self):
        return self.robot.compute_actions_dimension()

    def reset(self):
        # 重置环境

        if self.schedule is not None:
            steps = self.schedule["timesteps"]
            diffs = np.array(steps) - self.total_timesteps

            # 找到当前时间步所在的区间
            idx = len(diffs[diffs <= 0]) - 1

            self.num_cooperative = self.schedule["num_cooperative"][idx]
            self.num_non_cooperative = self.schedule["num_non_cooperative"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

            print("\n======== 训练计划 ========")
            print("协作机器人数量: ", self.num_cooperative)
            print("非协作机器人数量: ", self.num_non_cooperative)
            print("障碍物数量: ", self.num_obs)
            print("起点和终点的最小距离: ", self.min_start_goal_dis)
            print("======== 训练计划 ========\n")

        self.episode_timesteps = 0

        self.obstacles.clear()
        self.robots.clear()

        num_obs = self.num_obs
        robot_types = [True] * self.num_cooperative + [False] * self.num_non_cooperative
        assert len(robot_types) > 0, "机器人数量为0！"

        ##### 随机生成起点和终点的机器人
        num_robots = 0
        iteration = 500
        while True:
            start = self.rd.uniform(low=2.0 * np.ones(2), high=np.array([self.width - 2.0, self.height - 2.0]))
            goal = self.rd.uniform(low=2.0 * np.ones(2), high=np.array([self.width - 2.0, self.height - 2.0]))
            # print("start:", start, "goal:", goal)
            iteration -= 1
            if self.check_start_and_goal(start, goal):
                rob = robot.Robot(robot_types[num_robots])
                rob.start = start
                rob.goal = goal
                self.reset_robot(rob)
                self.robots.append(rob)
                num_robots += 1
            if iteration == 0 or num_robots == len(robot_types):
                break

        ##### 随机生成静态障碍物的位置和大小
        if num_obs > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low=5.0 * np.ones(2), high=np.array([self.width - 5.0, self.height - 5.0]))
                r = self.rd.uniform(low=self.obs_r_range[0], high=self.obs_r_range[1])
                obs = Obstacle(center[0], center[1], r)
                iteration -= 1
                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break

        return self.get_observations()

    def reset_robot(self, rob):
        # 重置机器人的状态
        rob.reach_goal = False
        rob.collision = False
        rob.deactivated = False
        rob.init_theta = self.rd.uniform(low=0.0, high=2 * np.pi)
        rob.init_speed = self.rd.uniform(low=0.0, high=rob.max_speed)
        rob.reset_state(current_velocity=np.zeros(2))

    def check_all_deactivated(self):
        # 检查是否所有机器人都已失活
        res = True
        for rob in self.robots:
            if not rob.deactivated:
                res = False
                break
        return res

    def check_all_reach_goal(self):
        # 检查是否所有机器人都已到达目标
        res = True
        for rob in self.robots:
            if not rob.reach_goal:
                res = False
                break
        return res

    def check_any_collision(self):
        # 检查是否发生了任何碰撞
        res = False
        for rob in self.robots:
            if rob.collision:
                res = True
                break
        return res

    def step(self, actions, coll_robots_list):
        rewards = [0] * len(self.robots)

        assert len(actions) == len(self.robots), "动作数量与机器人数量不一致！"
        assert self.check_all_reach_goal() is not True, "所有机器人已经到达目标，无可用动作！"

        # 为所有机器人执行动作
        for i, action in enumerate(actions):
            rob = self.robots[i]

            if rob.deactivated:
                # 如果机器人已经失活
                continue

            # 将动作保存到历史记录
            rob.action_history.append(action)

            dis_before = rob.dist_to_goal()

            # 执行动作后更新机器人的状态
            for _ in range(rob.N):
                rob.update_state(action, np.zeros(2))

            # 保存机器人的状态
            rob.trajectory.append([rob.x, rob.y, rob.theta, rob.speed, rob.velocity[0], rob.velocity[1]])

            dis_after = rob.dist_to_goal()

            # 每个时间步的固定惩罚
            rewards[i] += self.timestep_penalty

            # if i in coll_robots_list:
            #     rewards[i] += self.ttc_penalty

            # 为机器人靠近目标给予奖励
            rewards[i] += dis_before - dis_after

        # 获取所有机器人的观测值
        observations, collisions, reach_goals = self.get_observations()

        dones = [False] * len(self.robots)
        infos = [{"state": "normal"}] * len(self.robots)

        for idx, rob in enumerate(self.robots):
            if rob.deactivated:
                dones[idx] = True
                if rob.collision:
                    infos[idx] = {"state": "碰撞后失活"}
                elif rob.reach_goal:
                    infos[idx] = {"state": "到达目标后失活"}
                else:
                    raise RuntimeError("机器人失活只能由碰撞或到达目标引起！")
                continue

            if self.episode_timesteps >= 1000:
                dones[idx] = True
                infos[idx] = {"state": "时间步数过长"}
            elif collisions[idx]:
                rewards[idx] += self.collision_penalty
                dones[idx] = True
                infos[idx] = {"state": "碰撞"}
            elif reach_goals[idx]:
                rewards[idx] += self.goal_reward
                dones[idx] = True
                infos[idx] = {"state": "到达目标"}
            else:
                dones[idx] = False
                infos[idx] = {"state": "正常"}

        self.episode_timesteps += 1
        self.total_timesteps += 1

        return observations, rewards, dones, infos


    def get_global_observations(self):
        # 获取全局坐标系下所有机器人和障碍物的信息
        # print("self.robots:", self.robots, "self.obstacles:", self.obstacles)
        robots_matrix, obstacles_matrix = robot.Robot.get_global_states(self, self.robots, self.obstacles)

        return robots_matrix, obstacles_matrix

    def get_observations(self):
        # 获取所有机器人的观测值
        observations = []
        collisions = []
        reach_goals = []
        for robot in self.robots:
            observation, collision, reach_goal = robot.perception_output(self.obstacles, self.robots, self.observation_in_robot_frame)
            # print("observation:", observation, "collision:", collision, "reach_goal:", reach_goal)
            # print("len:", len(observation))
            observations.append(observation)
            collisions.append(collision)
            reach_goals.append(reach_goal)
        return observations, collisions, reach_goals

    def check_start_and_goal(self, start, goal):
        # 检查起点和终点是否满足约束条件

        # 起点和终点之间的距离足够远
        if np.linalg.norm(goal - start) < self.min_start_goal_dis:
            return False

        for robot in self.robots:
            dis_s = robot.start - start
            # 起点不能离已有机器人的起点太近
            if np.linalg.norm(dis_s) <= self.clear_r:
                return False

            dis_g = robot.goal - goal
            # 终点不能离已有机器人的终点太近
            if np.linalg.norm(dis_g) <= self.clear_r:
                return False

        return True

    def check_obstacle(self, obs):
        # 检查障碍物是否满足约束条件

        # 障碍物是否在地图范围内
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        for robot in self.robots:
            # 障碍物不能离机器人起点和终点太近
            obs_pos = np.array([obs.x, obs.y])
            dis_s = obs_pos - robot.start
            if np.linalg.norm(dis_s) < obs.r + self.clear_r:
                return False
            dis_g = obs_pos - robot.goal
            if np.linalg.norm(dis_g) < obs.r + self.clear_r:
                return False

        # 障碍物不能与其他障碍物重叠
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx * dx + dy * dy)

            if dis <= obstacle.r + obs.r:
                return False

        return True

    def reset_with_eval_config(self, eval_config):
        # 使用评估配置重置环境
        self.episode_timesteps = 0

        # 加载环境配置
        self.sd = eval_config["env"]["seed"]
        self.width = eval_config["env"]["width"]
        self.height = eval_config["env"]["height"]
        self.obs_r_range = copy.deepcopy(eval_config["env"]["obs_r_range"])
        self.clear_r = eval_config["env"]["clear_r"]
        self.timestep_penalty = eval_config["env"]["timestep_penalty"]
        self.collision_penalty = eval_config["env"]["collision_penalty"]
        self.goal_reward = eval_config["env"]["goal_reward"]

        # 加载障碍物
        self.obstacles.clear()
        for i in range(len(eval_config["env"]["obstacles"]["positions"])):
            center = eval_config["env"]["obstacles"]["positions"][i]
            r = eval_config["env"]["obstacles"]["r"][i]
            obs = Obstacle(center[0], center[1], r)
            self.obstacles.append(obs)

        # 加载机器人配置
        self.robots.clear()
        for i in range(len(eval_config["robots"]["cooperative"])):
            rob = robot.Robot(eval_config["robots"]["cooperative"][i])
            rob.dt = eval_config["robots"]["dt"][i]
            rob.N = eval_config["robots"]["N"][i]
            rob.length = eval_config["robots"]["length"][i]
            rob.width = eval_config["robots"]["width"][i]
            rob.r = eval_config["robots"]["r"][i]
            rob.detect_r = eval_config["robots"]["detect_r"][i]
            rob.goal_dis = eval_config["robots"]["goal_dis"][i]
            rob.obs_dis = eval_config["robots"]["obs_dis"][i]
            rob.max_speed = eval_config["robots"]["max_speed"][i]
            rob.a = np.array(eval_config["robots"]["a"][i])
            rob.w = np.array(eval_config["robots"]["w"][i])
            rob.start = np.array(eval_config["robots"]["start"][i])
            rob.goal = np.array(eval_config["robots"]["goal"][i])
            rob.compute_actions()
            rob.init_theta = eval_config["robots"]["init_theta"][i]
            rob.init_speed = eval_config["robots"]["init_speed"][i]

            rob.perception.range = eval_config["robots"]["perception"]["range"][i]
            rob.perception.angle = eval_config["robots"]["perception"]["angle"][i]

            rob.reset_state(current_velocity=np.zeros(2))

            self.robots.append(rob)

        return self.get_observations()

    def episode_data(self):
        # 保存episode数据
        episode = {}

        # 保存环境配置
        episode["env"] = {}
        episode["env"]["seed"] = self.sd
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward

        # 保存障碍物信息
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x, obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # 保存机器人信息
        episode["robots"] = {}
        episode["robots"]["cooperative"] = []
        episode["robots"]["dt"] = []
        episode["robots"]["N"] = []
        episode["robots"]["length"] = []
        episode["robots"]["width"] = []
        episode["robots"]["r"] = []
        episode["robots"]["detect_r"] = []
        episode["robots"]["goal_dis"] = []
        episode["robots"]["obs_dis"] = []
        episode["robots"]["max_speed"] = []
        episode["robots"]["a"] = []
        episode["robots"]["w"] = []
        episode["robots"]["start"] = []
        episode["robots"]["goal"] = []
        episode["robots"]["init_theta"] = []
        episode["robots"]["init_speed"] = []

        episode["robots"]["perception"] = {}
        episode["robots"]["perception"]["range"] = []
        episode["robots"]["perception"]["angle"] = []

        episode["robots"]["action_history"] = []
        episode["robots"]["trajectory"] = []

        for rob in self.robots:
            episode["robots"]["cooperative"].append(rob.cooperative)
            episode["robots"]["dt"].append(rob.dt)
            episode["robots"]["N"].append(rob.N)
            episode["robots"]["length"].append(rob.length)
            episode["robots"]["width"].append(rob.width)
            episode["robots"]["r"].append(rob.r)
            episode["robots"]["detect_r"].append(rob.detect_r)
            episode["robots"]["goal_dis"].append(rob.goal_dis)
            episode["robots"]["obs_dis"].append(rob.obs_dis)
            episode["robots"]["max_speed"].append(rob.max_speed)
            episode["robots"]["a"].append(list(rob.a))
            episode["robots"]["w"].append(list(rob.w))
            episode["robots"]["start"].append(list(rob.start))
            episode["robots"]["goal"].append(list(rob.goal))
            episode["robots"]["init_theta"].append(rob.init_theta)
            episode["robots"]["init_speed"].append(rob.init_speed)

            episode["robots"]["perception"]["range"].append(rob.perception.range)
            episode["robots"]["perception"]["angle"].append(rob.perception.angle)

            episode["robots"]["action_history"].append(copy.deepcopy(rob.action_history))
            episode["robots"]["trajectory"].append(copy.deepcopy(rob.trajectory))

        return episode

    def save_episode(self, filename):
        # 将episode数据保存为文件
        episode = self.episode_data()
        with open(filename, "w") as file:
            json.dump(episode, file)