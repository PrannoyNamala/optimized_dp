import os
import pandas as pd
import yaml
import numpy as np

import time

from pursuitevasion import PursuitEvasion, PotentialFieldEvader

from assignment_data import capture_1vs1, capture_2vs1, lp_solver, pur_ev_control1v1_1slice, pur_ev_control2v1_1slice
from odp.dynamics import PursuitEvasion1v1, PursuitEvasion2v1
from odp.Grid import Grid

import ray
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from typing import Dict

from ray.rllib.algorithms.ppo import PPO

from ray.rllib.policy import Policy

from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

import torch

scenario_name = "pursuit_evasion"

def env_creator(config: Dict):
    return PursuitEvasion(config["num_pursuers"], config["num_evaders"])

def get_rl_models():
    if not ray.is_initialized():
        ray.init()
        print("Ray init!")

    register_env(scenario_name, lambda config: env_creator(config))

    checkpoint_path = "/home/prannoy/pursuit_evasion_rllib/training_checkpoints/10v10_experiment"

    env = PursuitEvasion(4, 4)

    observations, dones = env.reset()

    register_env(scenario_name, lambda config: env_creator(config))
    # single_env = env_creator({'num_envs': 1, "scenario_config": {}})
    single_env = env_creator({'num_pursuers': 4, 'num_evaders': 4})
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    def gen_policy(i):
        return (None, obs_space[i+'_0'], act_space[i+'_0'], {})

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in ['pursuer', 'evader']
    }
    policy_ids = list(policies.keys())

    def policy_assign(agent_id, episode):
        if agent_id.split('_')[0]==("pursuer"):
            return "policy_pursuer"
        else:
            return "policy_evader"
        
    train_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            scenario_name,
            env_config =  {
                "num_pursuers": 10,
                "num_evaders": 10,
            })
        .reporting(
            keep_per_episode_custom_metrics=True,
        )
        .checkpointing(
            export_native_model_files=True
        )
        .training(
            lr=0.0001,
            lambda_= 0.95,
            kl_coeff = 0.1,
            vf_clip_param = 5.0,
            entropy_coeff = 0.01,
            train_batch_size = 4000,
            num_sgd_iter = 10,
            vf_share_layers = True,
        )
        .framework(framework="torch")
        .evaluation(
            evaluation_interval = 50,
            evaluation_duration=5,
            evaluation_duration_unit='episodes'
        )
        .env_runners(num_env_runners=1, batch_mode = "complete_episodes")
        .multi_agent(
            policies = policies,
            policy_mapping_fn = policy_assign,
        )
        
    )

    train_config = train_config.build()
    print("passed here")
    
    train_config.restore_from_path(checkpoint_path)

    
    pursuer_model = train_config.env_runner.module["policy_pursuer"]
    evader_model = train_config.env_runner.module["policy_evader"]
    
    return pursuer_model, evader_model

# Declare Number of Pursuers and Evaders Combo
#### Entered Manually ####
np_to_ne = [
            [1,4], [2,8], [3,12],
            [2,4], [3,6], [4,8], [5,10], [6,12],
            [5,5], [7,7], [9,9], [11,11], [13,13], 
            [4,2], [6,3], [8,4], [10,5], [12,6],
            [4,1], [8,2], [12,3]
            ]

episodes_per_combo = 100

episode_duration = 500

ev_controller = PotentialFieldEvader()

use_greedy_pursuer = True
use_rl_pursuer = False

use_rl_evader = False
use_potential_field_evader = True

pursuer_model, evader_model = get_rl_models()

# Create a folder named 'experiment_log'
if not os.path.exists('experiment_log'):
    os.makedirs('experiment_log')

# Add folder per the approach used by the agents
extra_tag= ""
if use_potential_field_evader:
    extra_tag += "_potential_field_evader"
elif use_rl_evader:
    extra_tag += "_rl_evader"
if use_greedy_pursuer:
    extra_tag += "_greedy_pursuer"
elif use_rl_pursuer:
    extra_tag += "_rl_pursuer"
else:
    extra_tag += "_assn_pursuer"
    
    
print(f"Extra Tag: {extra_tag}")
# date and time tag
timetag = time.strftime("_%Y%m%d_%H%M%S")

full_location = f"experiment_log/{timetag+extra_tag}"
    
os.makedirs(full_location, exist_ok=True)

car_1v1 = PursuitEvasion1v1(pursuer_R = 0.05, evader_R = 0.05)
car_2v1 = PursuitEvasion2v1(pursuer_R = 0.05, evader_R = 0.05)

grid_size  = 40
g_1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([grid_size, grid_size, grid_size, grid_size]), [])

grid_size  = 20

g_2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size]), [])


start_time = time.time()

for combo in np_to_ne:
    num_pursuer = combo[0]
    num_evader = combo[1]
    print(f"Running Number of Pursuers: {num_pursuer}, Number of Evaders: {num_evader}. Creating Folder")
    os.makedirs(f"{full_location}/{num_pursuer}P{num_evader}E", exist_ok=True)

    env = PursuitEvasion(num_pursuers=num_pursuer, num_evaders=num_evader)

    for i in range(episodes_per_combo):
        print(f"Running Episode {i+1}")
        # Make Episode Folder in n_pursuerPn_evader
        os.makedirs(f"{full_location}/{num_pursuer}P{num_evader}E/{i+1}", exist_ok=True)

        pur_names = [f"P{i}" for i in range(num_pursuer)]
        ev_names = [f"E{i}" for i in range(num_evader)]
        
        # Empty DataFrame to store agent positions
        agent_positions = pd.DataFrame(columns=pur_names+ev_names+["dones"]+["selections"])
        # Save Metrics in a Dictionary
        metrics = {"timesteps": 0, "ev_captures": 0, "episode_duration": 0, "Reason": "None"}

        observations, dones = env.reset()
        
        # Run for 500 timesteps
        episode_start_time = time.time()
        for t in range(episode_duration):
            current_evaders, current_pursuers = env.evaders, env.pursuers

            actions = [np.zeros((2,))]* (env.num_pursuers + env.num_evaders)

            Ic, one_v_one_value_list = capture_1vs1(current_pursuers, current_evaders, [])

            Pc, two_v_one_value_list = capture_2vs1(current_pursuers, current_evaders)

            selected, combo_order = lp_solver(one_v_one_value_list, two_v_one_value_list, num_pursuer, num_evader, dones)

            ### Add Logging Code Here ###
            agent_positions.loc[t] = [list(current_pursuers[i]) for i in range(env.num_pursuers)] + [list(current_evaders[j]) for j in range(env.num_evaders)] + [dones] + [selected]

            # Assign Random controls to pursuers and change it if assigned an evader
            actions[:env.num_pursuers] = [np.random.uniform(-1.0, 1.0, size=(2,))]*env.num_pursuers
            for j in range(len(selected)):
                if len(selected[j]) == 0:
                    pass
                elif selected[j][-1]<env.num_pursuers:
                    p_sel = selected[j][-1]
                    comb_state = (current_pursuers[p_sel][0], current_pursuers[p_sel][1], current_evaders[j][0], current_evaders[j][1])
                    pur_action, ev_action = pur_ev_control1v1_1slice(car_1v1, g_1v1, comb_state)
                    actions[p_sel] = pur_action if not (use_greedy_pursuer or use_rl_pursuer) else np.zeros((2,))
                    actions[env.num_pursuers+j] = ev_action if not (use_potential_field_evader or use_rl_evader) else np.zeros((2,))
                else:
                    p_1 = int(combo_order["combo"+str(selected[j][-1])+"ev"+str(j)].split(',')[0])
                    p_2 = int(combo_order["combo"+str(selected[j][-1])+"ev"+str(j)].split(',')[1])
                    comb_state = (current_pursuers[p_1][0], current_pursuers[p_1][1], current_pursuers[p_2][0], current_pursuers[p_2][1], current_evaders[j][0], current_evaders[j][1])
                    
                    p1_action, p2_action,ev_action = pur_ev_control2v1_1slice(car_2v1, g_2v1, comb_state)
                    actions[p_1] = p1_action if not (use_greedy_pursuer or use_rl_pursuer) else np.zeros((2,))
                    actions[p_2] = p2_action if not (use_greedy_pursuer or use_rl_pursuer) else np.zeros((2,))
                    actions[env.num_pursuers+j] = ev_action if not (use_potential_field_evader or use_rl_evader) else np.zeros((2,))
                    
            if use_potential_field_evader:
                active_pursuers_list = env.pursuers[~np.array(dones[:num_pursuer])]
        
                for ev in range(num_evader):
                    active_evader_bool = dones[num_pursuer:]
                    active_evader_bool[ev] = True
                    active_evaders_list = env.evaders[~np.array(active_evader_bool)]
                    entities_list = np.vstack((active_pursuers_list, active_evaders_list))
                    entities_radius_list = [0.05]*(entities_list.shape[0])
                    if not dones[num_pursuer+ev]:
                        actions[env.num_pursuers+ev] = ev_controller.compute_control(env.evaders[ev], entities_list, entities_radius_list)
            
            if use_greedy_pursuer:
                for p in range(num_pursuer):
                    if not dones[p]:
                        actions[p] = env.greedy_pursuer_action(p)
                        
            if use_rl_pursuer:
                for p in range(num_pursuer):
                    if not dones[p]:
                        actions[p] = pursuer_model.forward_inference({"obs": torch.tensor(observations[f"pursuer_{p}"], dtype=torch.float32)})
            
            if use_rl_evader:
                for e in range(num_evader):
                    if not dones[num_pursuer+e]:
                        actions[env.num_pursuers+e] = evader_model.forward_inference({"obs": torch.tensor(observations[f"evader_{e}"], dtype=torch.float32)})
            
            observations,_,dones,_,_ = env.step(actions)

            metrics["ev_captures"] = sum(dones[env.num_pursuers:])
            metrics["timesteps"] = t
            metrics["episode_duration"] = time.time() - episode_start_time
            metrics["Reason"] = "Time Out"

            if all(dones):
                print("Episode Completed. All agents done")
                metrics["Reason"] = "All Agents Done"
                metrics["ev_captures"] = env.num_evaders
                observations, dones = env.reset()
                break
            elif all(dones[env.num_pursuers:]):
                print("Episode Completed.Evaders All Done")
                metrics["Reason"] = "Evaders All Done"
                metrics["ev_captures"] = env.num_evaders
                observations, dones = env.reset()
                break
            elif all(dones[:env.num_pursuers]):
                print("Episode Completed. Pursuers All Done")
                metrics["Reason"] = "Pursuers All Done"
                observations, dones = env.reset()
                break

        
        # Save the Agent Positions to csv in the episode folder
        agent_positions.to_csv(f"{full_location}/{num_pursuer}P{num_evader}E/{i+1}/agent_positions.csv", index=False)
        # Save the Metrics to yaml in the episode folder
        with open(f"{full_location}/{num_pursuer}P{num_evader}E/{i+1}/metrics.yaml", 'w') as f:
            yaml.dump(metrics, f)


        print(f"Finished Combo {num_pursuer}P{num_evader}E")

end_time = time.time()

print("All Experiments Completed")
print(f"Total Time Taken: {end_time-start_time} seconds")
