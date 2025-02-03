import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pursuitevasion import PursuitEvasion
from matplotlib.animation import FuncAnimation
import pandas as pd
import re
import ast


# num_pursuer = 3
# num_evader = 1

# env = PursuitEvasion(num_pursuer, num_evader, 1, 0.05)

# with open('pur_pos.txt', 'r') as f:
#     pu_str = f.read()

# with open('ev_pos.txt', 'r') as f:
#     ev_str = f.read()

# pursuers_trajectory = json.loads(pu_str)

# evaders_trajectory = json.loads(ev_str)

# Extract num_pursuer and numimport ast_evader from folder name
folder_name = '1P4E'
match = re.match(r'(\d+)P(\d+)E', folder_name)
if match:
    num_pursuer = int(match.group(1))
    num_evader = int(match.group(2))
else:
    raise ValueError("Folder name does not match the expected pattern")

# Load DataFrame from CSV file
df = pd.read_csv(f'experiment_log/{folder_name}/1/agent_positions.csv').applymap(ast.literal_eval)
# print(df)  # Display the first few rows of the DataFrame

pursuer_names = [ f"P{i}" for i in range(num_pursuer)]
evader_names = [ f"E{i}" for i in range(num_evader)]

pursuers_trajectory = df[pursuer_names]
evaders_trajectory = df[evader_names]

pursuers_trajectory.columns = range(num_pursuer)
evaders_trajectory.columns = range(num_evader)

T = 30  # total simulation time T = [0.125s (25 A0 by D0), 0.165s (33 A2 by D2), 0.295s (59 A7 by D1),  0.340S (68 A4 by D3),  o.655s (131 A3 by D1), 0.720s (144 A6 by D0), 0.940s (188 A5 by D3), 1.455s (291 A1 by D3)]
deltat = 0.1 # calculation time interval
times = len(df)

env = PursuitEvasion(num_pursuers=num_pursuer, num_evaders=num_evader)


# times = len(pursuers_trajectory[0])

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')

# List to hold the patches and text objects
pursuer_circles = []
pursuer_labels = []
evader_circles = []
evader_labels = []

# Initialize the obstacle
obstacle = Circle([0, 0], radius=env.obstacle_radius, edgecolor='black', facecolor='none', linewidth=1)
ax.add_patch(obstacle)
ax.text(0, 0, "O", color="black", ha="center", va="center")  # Obstacle label

# Initialization function to set up objects
def init():
    global pursuer_circles, pursuer_labels, evader_circles, evader_labels

    # Initialize pursuers
    for i in range(num_pursuer):
        circle = Circle((0, 0), radius=env.agent_radius, edgecolor='blue', facecolor='none', linewidth=1)
        label = ax.text(0, 0, f"P{i}", color="blue", ha="center", va="center")
        pursuer_circles.append(circle)
        pursuer_labels.append(label)
        ax.add_patch(circle)

    # Initialize evaders
    for j in range(num_evader):
        circle = Circle((0, 0), radius=env.agent_radius, edgecolor='red', facecolor='none', linewidth=1)
        label = ax.text(0, 0, f"E{j}", color="red", ha="center", va="center")
        evader_circles.append(circle)
        evader_labels.append(label)
        ax.add_patch(circle)

    return pursuer_circles + evader_circles + pursuer_labels + evader_labels

# Update function for each frame
def update(t):
    # Update pursuers' positions
    for i in range(num_pursuer):
        x, y = pursuers_trajectory[i][t]
        pursuer_circles[i].center = (x, y)
        pursuer_labels[i].set_position((x, y))

    # Update evaders' positions
    for j in range(num_evader):
        x, y = evaders_trajectory[j][t]
        evader_circles[j].center = (x, y)
        evader_labels[j].set_position((x, y))

    return pursuer_circles + evader_circles + pursuer_labels + evader_labels

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, times), init_func=init, blit=True, interval=50)

# Display the animation
plt.show()

# Optionally save the animation as a video or GIF
ani.save("pursuit_evasion.gif", writer="pillow")

# for t in range(0, times+1):

#     ax.clear()
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     for i in range(num_pursuer):
#         circle = Circle(pursuers_trajectory[i][t], radius=env.agent_radius, edgecolor='blue', facecolor='none', linewidth=1)
#         ax.add_patch(circle)
#         ax.text(pursuers_trajectory[i][t][0], pursuers_trajectory[i][t][1], f"P{i}", color="blue", ha="center", va="center")

#     for j in range(num_evader):
#         circle = Circle(evaders_trajectory[j][t], radius=env.agent_radius, edgecolor='red', facecolor='none', linewidth=1)
#         ax.add_patch(circle)
#         ax.text(evaders_trajectory[j][t][0], evaders_trajectory[j][t][1], f"E{j}", color="red", ha="center", va="center")  # Optional text label

#     circle = Circle([0,0], radius=env.obstacle_radius, edgecolor='black', facecolor='none', linewidth=1)
#     ax.add_patch(circle)
#     ax.text(0, 0, "O", color="black", ha="center", va="center")  # Optional text label

#     fig.show()
#     plt.pause(0.01)

# fig.show()
