import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
from pursuitevasion import PursuitEvasion, PotentialFieldEvader
import numpy as np
from assignment_data import capture_1vs1, capture_2vs1, lp_solver


def plot_positions(file_path = '/home/prannoy/optimized_dp/Old_pe_logs/experiment_log_test_worked/7P7E/1/agent_positions.csv', timestep = 1, make_plot=False):

    data = pd.read_csv(file_path)

    positions = {}

    for column in data.columns[:-1]:

        positions[column] = eval(data.loc[timestep, column])



    if make_plot:
        # Create the plot with updated circle colors

        fig, ax = plt.subplots(figsize=(8, 8))



        # Define colors for Pursuer and Evaders

        circle_colors = {'P0': 'green', 'E0': 'red', 'E1': 'red', 'E2': 'red', 'E3': 'red'}



        # Plot points and draw circles

        for label, point in positions.items():

            ax.scatter(*point, label=label, s=100, edgecolor='black')

            circle = patches.Circle(point, 0.05, edgecolor=circle_colors[label], facecolor='none', linewidth=2, alpha=0.6)

            ax.add_patch(circle)

            ax.text(point[0], point[1], label, fontsize=12, ha='right', color=circle_colors[label])



        # Add a circle of radius 0.3 at the origin

        origin_circle = patches.Circle((0, 0), 0.3, edgecolor='black', facecolor='none', linewidth=2, linestyle='--', alpha=0.7)

        ax.add_patch(origin_circle)



        # Add the origin annotation

        ax.scatter(0, 0, color='black', label='Origin', s=100)

        ax.text(0, 0, 'Origin', fontsize=12, ha='right', color='black')



        # Set plot limits for better visualization

        ax.set_xlim(-1, 1)

        ax.set_ylim(-1, 1)



        # Add grid, legend, and titles

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.legend()

        ax.set_title('Positions of Pursuer (P0) and Evaders (E0, E1, E2, E3) with Radius Circles')

        ax.set_xlabel('X')

        ax.set_ylabel('Y')



        # Show the plot

        plt.show()

    return positions


if __name__ == '__main__':
    make_plot = False

    positions = plot_positions(make_plot=make_plot)

    if make_plot:
        raise SystemExit

    num_pursuer = 0
    num_evader = 0
    pursuers = []
    evaders = []
    for key in positions.keys():
        if key[0] == 'P':
            num_pursuer += 1
            pursuers.append(np.array(positions[key]))
        else:
            num_evader += 1
            evaders.append(np.array(positions[key]))

    env = PursuitEvasion(num_pursuers=num_pursuer, num_evaders=num_evader)

    env.pursuers = np.array(pursuers)
    env.evaders = np.array(evaders)

    _,_,dones,_,_ = env.step()

    current_evaders, current_pursuers = env.evaders, env.pursuers

    actions = [np.zeros((2,))]* (env.num_pursuers + env.num_evaders)

    Ic, one_v_one_value_list = capture_1vs1(current_pursuers, current_evaders, [])

    Pc, two_v_one_value_list = capture_2vs1(current_pursuers, current_evaders)

    selected, combo_order = lp_solver(one_v_one_value_list, two_v_one_value_list, num_pursuer, num_evader, dones)
    
    print(dones)
        
    ev_controller = PotentialFieldEvader()
    active_pursuers_list = env.pursuers[~np.array(dones[:num_pursuer])]
        
    for i in range(num_evader):
        active_evader_bool = dones[num_pursuer:]
        active_evader_bool[i] = True
        active_evaders_list = env.pursuers[~np.array(active_evader_bool)]
        entities_list = np.vstack((active_pursuers_list, active_evaders_list))
        entities_radius_list = [0.05]*(entities_list.shape[0])
        if not dones[num_pursuer+i]:
            print(f"Evader {i}", ev_controller.compute_control(env.evaders[i], entities_list, entities_radius_list))
