from pursuitevasion import PursuitEvasion, PotentialFieldEvader
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


num_pursuers=2
num_evaders=2

env = PursuitEvasion(num_pursuers=2, num_evaders=2)

ev_controller = PotentialFieldEvader()

_, dones = env.reset()

actions = [np.zeros((2,))]* (env.num_pursuers + env.num_evaders)

# Produce Evader actions
active_pursuers_list = env.pursuers[~np.array(dones[:num_pursuers])]

for ev in range(num_evaders):
    active_evader_bool = dones[num_pursuers:]
    active_evader_bool[ev] = True
    active_evaders_list = env.evaders[~np.array(active_evader_bool)]
    entities_list = np.vstack((active_pursuers_list, active_evaders_list))
    entities_radius_list = [0.05]*(entities_list.shape[0])
    if not dones[num_pursuers+ev]:
        actions[env.num_pursuers+ev] = ev_controller.compute_control(env.evaders[ev], entities_list, entities_radius_list)
        
        
# Plot all the entities with the actions for the evaders
# Initialize the environment
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize the obstacle
obstacle = Circle([0, 0], radius=env.obstacle_radius, edgecolor='black', facecolor='none', linewidth=1)
ax.add_patch(obstacle)
ax.text(0, 0, "O", color="black", ha="center", va="center")  # Obstacle label

# Initialize pursuers
pursuer_circles = []
pursuer_labels = []
for i in range(num_pursuers):
    pur_pos = env.pursuers[i]
    circle = Circle(pur_pos, radius=env.agent_radius, edgecolor='blue', facecolor='none', linewidth=1)
    label = ax.text(0, 0, f"P{i}", color="blue", ha="center", va="center")
    pursuer_circles.append(circle)
    pursuer_labels.append(label)
    ax.add_patch(circle)
    
# Initialize evaders
evader_circles = []
evader_labels = []
for j in range(num_evaders):
    ev_pos = env.evaders[j]
    circle = Circle(ev_pos, radius=env.agent_radius, edgecolor='red', facecolor='none', linewidth=1)
    label = ax.text(0, 0, f"E{j}", color="red", ha="center", va="center")
    evader_circles.append(circle)
    evader_labels.append(label)
    ax.add_patch(circle)
    
# Get arrows for the pursuer actions
arrows = []
for i in range(num_pursuers):
    pur_pos = env.pursuers[i]
    pur_action = actions[i]
    arrow = ax.arrow(pur_pos[0], pur_pos[1], pur_action[0], pur_action[1], head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    arrows.append(arrow)
    
# Get arrows for the evader actions
for j in range(num_evaders):
    ev_pos = env.evaders[j]
    ev_action = actions[env.num_pursuers+j]
    arrow = ax.arrow(ev_pos[0], ev_pos[1], ev_action[0], ev_action[1], head_width=0.05, head_length=0.1, fc='red', ec='red')
    arrows.append(arrow)
    
# Set x-lim and y-lim
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()

    
