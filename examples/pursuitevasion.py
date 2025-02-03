from typing import List
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ray.rllib.env import MultiAgentEnv

class PursuitEvasion(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_pursuers, num_evaders, num_obstacles=1, agent_radius=0.1, obstacle_radius=0.4, agent_vel=0.2, max_timesteps = 500):
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.n_agents = self.num_pursuers+self.num_evaders
        self.num_obstacles = num_obstacles
        self.agent_radius = agent_radius
        self.obstacle_radius = obstacle_radius
        self.pursuers = np.random.rand(num_pursuers, 2) - 1
        self.evaders = np.random.rand(num_pursuers, 2) - 1
        self.obstacles = np.random.rand(num_obstacles, 2) - 1
        self.velocities = np.zeros((num_pursuers + num_evaders, 2))
        self.agent_vel = agent_vel
        self.active_agents = np.ones(num_pursuers + num_evaders, dtype=bool)
        
        self.possible_agents = ["pursuer", "evader"]
        
        self.timesteps = 0
        
        self.max_timesteps = max_timesteps

        self._agent_ids = [f"pursuer_{i}" for i in range(num_pursuers)]+[f"evader_{i}" for i in range(num_evaders)]
        self.done_once = {agent_id: False for agent_id in self._agent_ids}
        
        self.agents = self._agent_ids


        # self.action_space = spaces.Tuple([spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for _ in range(self.n_agents)])
        # self.observation_space = spaces.Tuple([spaces.Box(low=-1.0, high=1.0, shape=(8+num_obstacles+1,), dtype=np.float32) for _ in range(self.n_agents)])
        
        self.action_space = spaces.Dict({agent_id: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for agent_id in self._agent_ids})
        
        self.observation_space = spaces.Dict({agent_id: spaces.Box(low=-1.5, high=1.5, shape=(8+num_obstacles+1,), dtype=np.float32) for agent_id in self._agent_ids})

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        

    def _get_obs(self):
        obs = np.concatenate([self.pursuers.flatten(), self.evaders.flatten(), self.obstacles.flatten()])
        return obs

    def reset(self, seed=None, options=None):

        points = []
        while len(points) < self.num_pursuers:
            # Randomly generate x and y within [-1, 1]
            x, y = np.random.uniform(-(1 - 2*self.agent_radius), (1 - 2*self.agent_radius)), np.random.uniform(-(1 - 2*self.agent_radius), (1 - 2*self.agent_radius))
            
            # Check if the point is outside the circle of radius 0.35
            if np.sqrt(x**2 + y**2) > (self.obstacle_radius+2*self.agent_radius):
                points.append((x, y))

        self.pursuers = np.array(points)

        points = []
        while len(points) < self.num_evaders:
            # Randomly generate x and y within [-1, 1]
            x, y = np.random.uniform(-(1 - 2*self.agent_radius), (1 - 2*self.agent_radius)), np.random.uniform(-(1 - 2*self.agent_radius), (1 - 2*self.agent_radius))
            
            # Check if the point is outside the circle of radius 0.35
            if np.sqrt(x**2 + y**2) > (self.obstacle_radius+2*self.agent_radius):
                points.append((x, y))

        self.evaders = np.array(points)

        self.obstacles = np.array([0,0])
        self.velocities = np.zeros((self.num_pursuers + self.num_evaders, 2))
        self.active_agents = np.ones(self.num_pursuers + self.num_evaders, dtype=bool)
        
        self.done_once = {agent_id: False for agent_id in self._agent_ids}
        
        observations_values = []
        for i in range(self.num_pursuers+self.num_evaders):
            _observation = self._get_individual_obs(i)
            observations_values.append(_observation)

        _, _, dones, _, _= self.step()
        
        self.timesteps = 0

        return dict(zip(self._agent_ids, observations_values)), dict(zip(self._agent_ids, dones)) #, []*(self.num_pursuers + self.num_evaders), [False]*(self.num_pursuers + self.num_evaders)

    def _get_individual_obs(self, agent_index):
        # Define individual observation space
        if agent_index < self.num_pursuers:
            # Pursuer observation
            observation = self._get_pursuer_obs(agent_index)
        else:
            # Evader observation
            observation = self._get_evader_obs(agent_index)

        return observation if self.active_agents[agent_index] else np.random.uniform(-1.5, -1.5, observation.shape)


    def _get_pursuer_obs(self, agent_index):
        # Pursuer observation: 
        #   1. Position (2D)
        #   2. Velocity (2D)
        #   3. Nearest evader's position (2D)
        #   4. Distance to nearest obstacle
        #   5. Distance to boundary
        nearest_evader_dist = np.inf
        nearest_evader_pos = np.zeros(2)
        
        for i in range(self.num_evaders):
            dist = np.linalg.norm(self.pursuers[agent_index] - self.evaders[i])
            if dist < nearest_evader_dist:
                nearest_evader_dist = dist
                nearest_evader_pos = self.evaders[i]

        nearest_pursuer_dist = np.inf
        nearest_pursuer_pos = np.zeros(2)

        for i in range(self.num_pursuers):
            if agent_index == i:
                continue
            dist = np.linalg.norm(self.pursuers[agent_index] - self.pursuers[i])
            if dist < nearest_pursuer_dist:
                nearest_pursuer_dist = dist
                nearest_pursuer_pos = self.pursuers[i]

        obstacle_dist = self._get_obstacle_dist(self.pursuers[agent_index])
        boundary_dist = self._get_boundary_dist(self.pursuers[agent_index])

        observation = np.concatenate([self.pursuers[agent_index], 
                                    self.velocities[agent_index], 
                                    nearest_evader_pos,
                                    nearest_pursuer_pos, 
                                    [obstacle_dist], 
                                    [boundary_dist]])

        return observation
    

    def _get_evader_obs(self, agent_index):
        # Evader observation: 
        #   1. Position (2D)
        #   2. Velocity (2D)
        #   3. Nearest pursuer's position (2D)
        #   3. Nearest evaders's position (2D)
        #   4. Distance to nearest obstacle
        #   5. Distance to boundary
        nearest_pursuer_dist = np.inf
        nearest_pursuer_pos = np.zeros(2)
        
        for i in range(self.num_pursuers):
            dist = np.linalg.norm(self.evaders[agent_index - self.num_pursuers] - self.pursuers[i])
            if dist < nearest_pursuer_dist:
                nearest_pursuer_dist = dist
                nearest_pursuer_pos = self.pursuers[i]

        nearest_evader_dist = np.inf
        nearest_evader_pos = np.zeros(2)

        for i in range(self.num_evaders):
            if agent_index - self.num_pursuers == i:
                continue
            dist = np.linalg.norm(self.evaders[agent_index - self.num_pursuers] - self.evaders[i])
            if dist < nearest_evader_dist:
                nearest_evader_dist = dist
                nearest_evader_pos = self.evaders[i]

        obstacle_dist = self._get_obstacle_dist(self.evaders[agent_index - self.num_pursuers])
        boundary_dist = self._get_boundary_dist(self.evaders[agent_index - self.num_pursuers])

        observation = np.concatenate([self.evaders[agent_index - self.num_pursuers], 
                                    self.velocities[agent_index], 
                                    nearest_pursuer_pos,
                                    nearest_evader_pos, 
                                    [obstacle_dist], 
                                    [boundary_dist]])

        return observation


    def _get_obstacle_dist(self, position):
        # Calculate distance to nearest obstacle
        obstacle_dist = np.inf
        for obstacle in self.obstacles:
            dist = np.linalg.norm(position - obstacle) - self.obstacle_radius
            if dist < obstacle_dist:
                obstacle_dist = dist
        return obstacle_dist


    def _get_boundary_dist(self, position):
        # Calculate distance to boundary
        boundary_dist = np.min([np.abs(position[0] - 1), np.abs(position[0] + 1), 
                                np.abs(position[1] - 1), np.abs(position[1] + 1)])
        return boundary_dist
               
    def get_observation_space(self, _):
        return self.observation_space['pursuer_0']
    
    def agent_id_frm_idx(self, agent_index):
        if agent_index<self.num_pursuers:
            return self._agent_ids[agent_index]
        elif agent_index>=self.num_pursuers:
            return self._agent_ids[agent_index-self.num_pursuers]
        else:
            raise ValueError("Invalid Index")
    
    def get_action_space(self, _):
        return self.action_space['pursuer_0']

    def step(self, actions_dict = None, dt=0.1):
        # print(actions_dict)
        if actions_dict is None:
            actions = [np.array((2,))]*(self.num_pursuers+self.num_evaders)
        else:
            actions = []
            for i in self._agent_ids:
                actions.append(actions_dict.get(i, np.array((2,))))
        rewards = np.zeros(self.num_pursuers + self.num_evaders)
        # done = [False]*(self.num_pursuers + self.num_evaders)
        observations = []

        # Update velocities
        for i in range(self.num_pursuers + self.num_evaders):
            if self.active_agents[i]:
                self.velocities[i] = self.agent_vel*actions[i]

        # Update positions
        for i in range(self.num_pursuers + self.num_evaders):
            if self.active_agents[i]:
                if i < self.num_pursuers:
                    self.pursuers[i] += self.velocities[i] * dt

                    # Boundary check
                    if (self.pursuers[i, 0] < -(1 - self.agent_radius) or self.pursuers[i, 0] > (1 - self.agent_radius) or
                        self.pursuers[i, 1] < -(1 - self.agent_radius) or self.pursuers[i, 1] > (1 - self.agent_radius)) and i < self.num_pursuers:
                        self.active_agents[i] = False
                        rewards[i] = -10
                        self.done_once[self.agent_id_frm_idx(i)] = True

                    # Obstacle collision check
                    for obstacle in self.obstacles:
                        dist_pursuer = np.linalg.norm(self.pursuers[i] - obstacle) if i < self.num_pursuers else np.inf
                        if dist_pursuer < self.agent_radius + self.obstacle_radius:
                            self.active_agents[i] = False
                            rewards[i] = -10
                            self.done_once[self.agent_id_frm_idx(i)] = True

                else:
                    self.evaders[i - self.num_pursuers] += self.velocities[i] * dt

                    # Boundary check
                    if (self.evaders[i - self.num_pursuers, 0] < -(1 - self.agent_radius) or self.evaders[i - self.num_pursuers, 0] > (1 - self.agent_radius) or
                        self.evaders[i - self.num_pursuers, 1] < -(1 - self.agent_radius) or self.evaders[i - self.num_pursuers, 1] > (1 - self.agent_radius)) and i >= self.num_pursuers:
                        self.active_agents[i] = False
                        rewards[i] = -10
                        self.done_once[self.agent_id_frm_idx(i)] = True

                    # Obstacle collision check
                    for obstacle in self.obstacles:
                        dist_evader = np.linalg.norm(self.evaders[i - self.num_pursuers] - obstacle) if i >= self.num_pursuers else np.inf
                        if dist_evader < self.agent_radius + self.obstacle_radius:
                            self.active_agents[i] = False
                            rewards[i] = -10
                            self.done_once[self.agent_id_frm_idx(i)] = True

                # Inter-team collision check
                for j in range(self.num_pursuers + self.num_evaders):
                    if i != j and self.active_agents[i] and self.active_agents[j]:
                        dist_pursuer_evader = np.linalg.norm(self.pursuers[i] - self.evaders[j - self.num_pursuers]) if i < self.num_pursuers and j >= self.num_pursuers else np.inf
                        dist_pursuer_pursuer = np.linalg.norm(self.pursuers[i] - self.pursuers[j]) if i < self.num_pursuers and j < self.num_pursuers else np.inf
                        dist_evader_evader = np.linalg.norm(self.evaders[i - self.num_pursuers] - self.evaders[j - self.num_pursuers]) if i >= self.num_pursuers and j >= self.num_pursuers else np.inf
                        if dist_pursuer_evader < self.agent_radius + self.agent_radius:
                            self.active_agents[j] = False
                            rewards[j] = -20
                            self.done_once[self.agent_id_frm_idx(j)] = True

                            self.active_agents[i] = False
                            rewards[i] = 20
                            self.done_once[self.agent_id_frm_idx(i)] = True
                            
                        elif dist_pursuer_pursuer < self.agent_radius + self.agent_radius:
                            self.active_agents[i] = False
                            rewards[i] = -20
                            self.done_once[self.agent_id_frm_idx(i)] = True
                            
                        elif dist_evader_evader < self.agent_radius + self.agent_radius:
                            self.active_agents[i] = False
                            rewards[i] = -20
                            self.done_once[self.agent_id_frm_idx(i)] = True

            # Get individual observation
            observation = self._get_individual_obs(i)
            observations.append(observation)
            
        # done = {agent_id: not self.active_agents[i] for i, agent_id in enumerate(self._agent_ids)}
        done = {}
        
        for i, agent_id in enumerate(self._agent_ids):
            done[agent_id] = not self.active_agents[i] if self.done_once else self.active_agents[i]
        
        done["__all__"] = (self.timesteps >= self.max_timesteps) or (np.sum(self.active_agents) == 0) or sum(self.done_once.values())
        
        
        
        for agent_id, terminated in done.items():
            if agent_id != "__all__" and terminated:
                self.active_agents[self._agent_ids.index(agent_id)] = False

        truncated = {agent_id: self.timesteps >= self.max_timesteps for agent_id in self._agent_ids}
        truncated["__all__"] = self.timesteps >= self.max_timesteps
            
        self.timesteps += 1
        
        # print("Step Debugging:")
        # print("Observations:", dict(zip(self._agent_ids, observations)))
        # print("Rewards:", dict(zip(self._agent_ids, rewards)))
        # print("Done:", done)
        # print("Truncated:", truncated)


        return (
            dict(zip(self._agent_ids, observations)), 
            dict(zip(self._agent_ids, rewards)), 
            done,
            truncated,
            {}
        )

    def render(self, mode='human'):
        

        self.ax.clear()

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        
        for i in range(self.num_pursuers):
            circle = Circle(self.pursuers[i,:], radius=self.agent_radius, edgecolor='blue', facecolor='none', linewidth=1)
            self.ax.add_patch(circle)
            self.ax.text(self.pursuers[i,0], self.pursuers[i,1], "P", color="blue", ha="center", va="center")

        for i in range(self.num_evaders):
            circle = Circle(self.evaders[i,:], radius=self.agent_radius, edgecolor='red', facecolor='none', linewidth=1)
            self.ax.add_patch(circle)
            self.ax.text(self.evaders[i,0], self.evaders[i,1], "E", color="red", ha="center", va="center")  # Optional text label

        circle = Circle([0,0], radius=self.obstacle_radius, edgecolor='black', facecolor='none', linewidth=1)
        self.ax.add_patch(circle)
        self.ax.text(0, 0, "O", color="black", ha="center", va="center")  # Optional text label

        plt.pause(0.1)  # Short pause to display the plot

        print("Press 'q' to continue...")
        while True:
            key = input()  # Wait for input
            if key.lower() == "q":
                break  # Continue simulation if 'q' is pressed

        
        
def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.zeros_like(vector)
    return vector / magnitude

class PotentialFieldEvader:
    def __init__(self, vel_agent=0.2, agent_radius=0.05, boundary=[-1,1,-1,1]):
        self.vel_agent = vel_agent
        self.agent_radius = agent_radius
        self.boundary = boundary

    def compute_repulsive_force(self, agent_position, entities: List, entity_radius: List):
        force = np.zeros(2)
        for entity, radius in zip(entities, entity_radius):
            diff = agent_position - np.array(entity)
            distance = np.linalg.norm(diff)
            effective_distance = distance - self.agent_radius - radius
            if effective_distance < 0:
                effective_distance = 1e-6  # Avoid division by zero or overlap
            direction = normalize(diff)
            force += direction / (effective_distance**2)
        return force
    
    def compute_boundary_force(self, agent_position):
        force = np.zeros(2)
        x, y = agent_position

        if x - self.agent_radius < self.boundary[0]:
            force[0] += 1 / ((x - self.boundary[0] - self.agent_radius)**2)
        if x + self.agent_radius > self.boundary[1]:
            force[0] -= 1 / ((self.boundary[1] - x - self.agent_radius)**2)

        if y - self.agent_radius < self.boundary[2]:
            force[1] += 1 / ((y - self.boundary[2] - self.agent_radius)**2)
        if y + self.agent_radius > self.boundary[3]:
            force[1] -= 1 / ((self.boundary[3] - y - self.agent_radius)**2)

        return force

    def compute_obstacle_force(self, agent_position, obs_pos=np.array((0, 0)), obs_radius=0.3):
        diff = agent_position - obs_pos
        distance = np.linalg.norm(diff)
        effective_distance = distance - self.agent_radius - obs_radius
        if effective_distance < 0:
            effective_distance = 1e-6  # Avoid division by zero or overlap
        direction = normalize(diff)
        force = direction / (effective_distance**2)
        return force

    def compute_control(self, agent_position, entities: List, entity_radius: List, obs_pos=np.array((0, 0)), obs_radius=0.3):
        repulsive_force = self.compute_repulsive_force(agent_position, entities, entity_radius)
        boundary_force = self.compute_boundary_force(agent_position)
        obstacle_force = self.compute_obstacle_force(agent_position, obs_pos, obs_radius)

        total_force = repulsive_force + boundary_force + obstacle_force
        control = normalize(total_force) if np.linalg.norm(total_force) > 0 else np.zeros(2)
        return control

