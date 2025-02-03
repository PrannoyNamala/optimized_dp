import numpy as np
import math
import time
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface
from odp.Grid import Grid
import datetime
from mip import *
from pursuitevasion import PursuitEvasion
from itertools import combinations
import pandas as pd
from odp.dynamics import PursuitEvasion1v1, PursuitEvasion2v1
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json


def spa_deriv(index, V, g, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension
    From Michael
    Args:
        index: (a1x, a1y)
        V (ndarray): [..., neg2pos] where neg2pos is a list [scalar] or []
        g (class): the instance of the corresponding Grid
        periodic_dims (list): the corrsponding periodical dimensions []

    Returns:
        List of left and right spatial derivatives for each dimension
    """
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1:])

        next_index = tuple(
            left_index + [index[dim] + 1] + right_index
        )
        prev_index = tuple(
            left_index + [index[dim] - 1] + right_index
        )

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [V.shape[dim] - 1] + right_index
                )
                left_boundary = V[left_periodic_boundary_index]
            else:
                left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(V[index])
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(
                    left_index + [0] + right_index
                )
                right_boundary = V[right_periodic_boundary_index]
            else:
                right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign([V[index]])
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append(((left_deriv + right_deriv) / 2))
    return spa_derivatives

def po2slice1vs1(attacker, defender, grid_size):
    """ Convert the position of the attacker and defender to the slice of the value function for 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid

    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker[0], attacker[1], defender[0], defender[1])  # (xA1, yA1, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)

# check in the current state, the pursuer is captured by the evader or not
def check_1vs1(attacker, defender, value1vs1):
    """ Check if the attacker could escape from the defender in a 1 vs 1 game.

    Args:
        attacker (np.ndarray): the attacker's state
        defender (np.ndarray): the defender's state
        value1vs1 (np.ndarray): the value function for 1 vs 1 game

    Returns:
        bool: False, if the attacker could escape (the attacker will win)
    """
    joint_slice = po2slice1vs1(attacker, defender, value1vs1.shape[0])

    return value1vs1[joint_slice] > 0, value1vs1[joint_slice] 

def capture_1vs1(pursuers, evaders, stops):
    """ Returns a list Ic that contains all pursuers that the evader couldn't capture, [[a1, a3], ...]

    Args:
        pursuers (list): positions (set) of all pursuers, [(a1x, a1y), ...]
        evaders (list): positions (set) of all evaders, [(d1x, d1y), ...]
        value1v1 (ndarray): 1v1 HJ value function
        stops (list): the captured pursuers index
    """

    value1v1 = np.load('1v1pe_g_40_speed0.2_circle0.4.npy') # Update

    num_pursuer, num_evader = len(pursuers), len(evaders)
    Ic = []
    values = []
    # generate I
    for i in range(num_evader):
        Ic.append([])
        values.append([])
        eix, eiy = evaders[i]
        
        for j in range(num_pursuer):
            if i in stops:  # ignore captured pursuers
                Ic[j].append(i)
            else:
                pjx, pjy = pursuers[j]
                flag, val = check_1vs1((pjx, pjy), (eix, eiy), value1v1)
                if not flag:  # evader j could not capture pursuer i
                    Ic[i].append(j)
                values[i].append(val)
    return Ic, values

# generate the capture pair list P and the capture pair complement list Pc
def capture_2vs1(pursuers, evaders):
    """ Returns a list Pc that contains all pairs of attackers that the defender couldn't capture, [[(a1, a2), (a2, a3)], ...]

    Args:
        attackers (list): positions (set) of all attackers, [(a1x, a1y), ...]
        defenders (list): positions (set) of all defenders, [(d1x, d1y), ...]
        value2v1 (ndarray): 2v1 HJ value function [, , , , ,]
    """
    value2v1 = np.load('2v1pe_g_20_speed0.2_circle0.4.npy')
    num_pursuers, num_evaders = len(pursuers), len(evaders)
    Pc = []
    values = {}
    # generate Pc
    for k in range(num_evaders):
        Pc.append([])
        values["ev"+str(k)] = []
        ekx, eky = evaders[k]
        for i in range(num_pursuers):
            pix, piy = pursuers[i]
            for j in range(num_pursuers):
                if i==j:
                  # values[j][i].append("_")
                  continue
                pjx, pjy = pursuers[j]
                
                flag, val = check_2vs1((pjx, pjy), (pix, piy), (ekx, eky), value2v1)
                if not flag:
                    Pc[k].append((i, j))
                values["ev"+str(k)].append((i,j, val))
    return Pc, values

def check_2vs1(attacker_i, attacker_k, defender, value2vs1):
    """ Check if the attackers could escape from the defender in a 2 vs 1 game.
    Here escape means that at least one of the attackers could escape from the defender.

    Args:
        attacker_i (np.ndarray): the attacker_i's states
        attacker_j (np.ndarray): the attacker_j's states
        defender (np.ndarray): the defender's state
        value2vs1 (np.ndarray): the value function for 2 vs 1 game

    Returns:
        bool: False, if the attackers could escape (the attackers will win)
    """
    joint_slice = po2slice2vs1(attacker_i, attacker_k, defender, value2vs1.shape[0])

    return value2vs1[joint_slice] > 0, value2vs1[joint_slice]

def po2slice2vs1(attacker_i, attacker_k, defender, grid_size):
    """ Convert the position of the attackers and defender to the slice of the value function for 2 vs 1 game.

    Args:
        attackers (np.ndarray): the attackers' states
        defender (np.ndarray): the defender's state
        grid_size (int): the size of the grid

    Returns:
        joint_slice (tuple): the joint slice of the joint state using the grid size

    """
    joint_state = (attacker_i[0], attacker_i[1], attacker_k[0], attacker_k[1], defender[0], defender[1])  # (xA1, yA1, xA2, yA2, xD1, yD1)
    joint_slice = []
    grid_points = np.linspace(-1, +1, num=grid_size)
    for i, s in enumerate(joint_state):
        idx = np.searchsorted(grid_points, s)
        if idx > 0 and (
            idx == len(grid_points)
            or math.fabs(s - grid_points[idx - 1])
            < math.fabs(s - grid_points[idx])
        ):
            joint_slice.append(idx - 1)
        else:
            joint_slice.append(idx)

    return tuple(joint_slice)

def pur_ev_control1v1_1slice(agents_1v1, grid1v1, jointstate1v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid1v1 (class): the corresponding Grid instance
    value1v1 (ndarray): 1v1 HJ reachability value function with only final slice
    agents_1v1 (class): the corresponding AttackerDefender instance
    joint_states1v1 (tuple): the corresponding positions of (A1, D1)
    """
    # calculate the derivatives
    # v = value1v1[...] # Minh: v = value1v0[..., neg2pos[0]]
    value1v1 = np.load('1v1pe_g_40_speed0.2_circle0.4.npy')
    start_time = datetime.datetime.now()
    # print(f"The shape of the input value1v1 of defender is {value1v1.shape}. \n")
    spat_deriv_vector = spa_deriv(grid1v1.get_index(jointstate1v1), value1v1, grid1v1)
    opt_d1, opt_d2 = agents_1v1.optDstb_inPython(spat_deriv_vector)
    opt_u1, opt_u2 = agents_1v1.optCtrl_inPython(spat_deriv_vector)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 4D spatial derivative vector is {end_time-start_time}. \n")
    return np.array((opt_u1, opt_u2)), np.array((opt_d1, opt_d2))

def pur_ev_control2v1_1slice(agents_2v1, grid2v1, jointstate2v1):
    """Return a list of 2-dimensional control inputs of one defender based on the value function
    
    Args:
    grid2v1 (class): the corresponding Grid instance
    value2v1 (ndarray): 1v1 HJ reachability value function with only final slice
    agents_2v1 (class): the corresponding AttackerDefender instance
    joint_states2v1 (tuple): the corresponding positions of (A1, A2, D)
    """
    # calculate the derivatives
    start_time = datetime.datetime.now()
    value2v1 = np.load('2v1pe_g_20_speed0.2_circle0.4.npy')
    # print(f"The shape of the input value2v1 of defender is {value2v1.shape}. \n")
    spat_deriv_vector = spa_deriv(grid2v1.get_index(jointstate2v1), value2v1, grid2v1)
    opt_u1, opt_u2, opt_u3, opt_u4 = agents_2v1.optCtrl_inPython(spat_deriv_vector)
    opt_d1, opt_d2 = agents_2v1.optDstb_inPython(spat_deriv_vector)
    end_time = datetime.datetime.now()
    # print(f"The calculation of 6D spatial derivative vector is {end_time-start_time}. \n")
    return np.array((opt_u1, opt_u2)), np.array((opt_u3, opt_u4)), np.array((opt_d1, opt_d2))

# set up and solve the mixed integer programming question
def mip_solver(num_pursuer, num_evader, Pc, Ic):
    """ Returns a list selected that contains all allocated attackers that the defender could capture, [[a1, a3], ...]

    Args:
        num_attackers (int): the number of attackers
        num_defenders (int): the number of defenders
        Pc (list): constraint pairs of attackers of every defender
        Ic (list): constraint individual attacker of every defender
    """
    # initialize the solver
    model = Model(solver_name=CBC) # use GRB for Gurobi, CBC default
    e = [[model.add_var(var_type=BINARY) for i in range(num_pursuer)] for j in range(num_evader)] # e[attacker index][defender index]
    
    # add constraints
    # Atmost two pursuers assigned to an evader
    for j in range(num_evader):
        model += xsum(e[i][j] for i in range(num_pursuer)) <= 2
    # One pursuer can capture only one evader
    for i in range(num_pursuer):
        model += xsum(e[i][j] for j in range(num_evader)) <= 1
    # add constraints 12c Pc
    for j in range(num_evader):
        for pairs in (Pc[j]):
            # print(pairs)
            model += e[pairs[0]][j] + e[pairs[1]][j] >= 1
            model += e[pairs[0]][j] + e[pairs[1]][j] <= 2
    # add constraints 12f Ic
    for j in range(num_evader):
        for indiv in (Ic[j]):
            # print(indiv)
            model += e[indiv][j] <= 1
    # set up objective functions
    model.objective = maximize(xsum(e[i][j] for j in range(num_evader) for i in range(num_pursuer)))
    # problem solving
    model.max_gap = 0.05
    # log_status = []
    status = model.optimize(max_seconds=300)
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(model.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {} '.format(model.objective_value, model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {} '.format(model.objective_bound))
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        print('Solution:')
        selected = []
        for j in range(num_evader):
            selected.append([])
            for i in range(num_pursuer):
                if e[i][j].x >= 0.9:
                    selected[j].append(i)
        print(selected)
    return selected

def lp_solver(one_v_one_value_list, two_v_one_value_list, num_pursuers, num_evaders, dones):
    # Generate all unique combinations of pursuers
    num_pursuers_combo = len(list(combinations(range(1, num_pursuers + 1), 2)))

    # Map combinations to indices
    combo_to_pursuers = {num_pursuers + idx: list(c) for idx, c in enumerate(combinations(range(num_pursuers), 2))}
    
    # Subsets of Combo based on Individual Pursuer
    combo_to_pursuer_subsets = {i:[] for i in range(num_pursuers)}
    for keys_ in combo_to_pursuers.keys():
        for indices_ in combo_to_pursuers[keys_]:
            combo_to_pursuer_subsets[indices_].append(keys_)
    
    # Generate random data for the cost matrix
    data_ = np.zeros((num_pursuers + num_pursuers_combo, num_evaders))

    # Update 1v1 data
    data_[:num_pursuers, :num_evaders] = np.array(one_v_one_value_list).T

    # Making a 2v1 dataframe
    df_data = []
    for ev_key in two_v_one_value_list.keys():
        list_ = two_v_one_value_list[ev_key]
        for item_ in list_:
            df_data.append([int(ev_key[-1])] + list(item_))

    twovone_df = pd.DataFrame(data=df_data, columns = ["evader", "p1", "p2", "value"])

    active_pursuers = num_pursuers
    active_evaders = num_evaders

    # Update 2v1
    combo_order = {}
    for ev in range(num_evaders):
        for combo_id in range(num_pursuers, num_pursuers + num_pursuers_combo):
            pur_selection = combo_to_pursuers[combo_id]
            val_n = twovone_df.query(f'evader == {ev} and p1 == {pur_selection[0]} and p2 == {pur_selection[1]}')
            val_r = twovone_df.query(f'evader == {ev} and p1 == {pur_selection[1]} and p2 == {pur_selection[0]}')

            if val_n.empty or val_r.empty:
                data_[combo_id, ev] = np.inf
                combo_order["combo"+str(combo_id)+"ev"+str(ev)] = str(pur_selection[1])+','+str(pur_selection[0])
            else:
                data_[combo_id, ev] = min(val_n["value"].values[-1], val_r["value"].values[-1])
                combo_order["combo"+str(combo_id)+"ev"+str(ev)] = str(pur_selection[0])+','+str(pur_selection[1]) if val_n["value"].values[-1] <= val_r["value"].values[-1] else str(pur_selection[1])+','+str(pur_selection[0])

    # Create the assignment model
    assn_model = Model(sense=MINIMIZE)

    # Define binary decision variables e[i][j]
    e = [[assn_model.add_var(var_type=BINARY) for j in range(num_evaders)] for i in range(num_pursuers + num_pursuers_combo)]

    # Define the objective function
    assn_model.objective = xsum(e[i][j] * data_[i][j] for i in range(num_pursuers + num_pursuers_combo) for j in range(num_evaders)) # (3 if i < num_pursuers else 1) * 

    # Constraint: Assignement shouldn't happen if the agent is terminated
    for (a,k) in enumerate(dones):
        if k:
            if a < num_pursuers:
                # Pursuer "a" is Done. No individual or dual assignments for that evader
                # No individual Assignment
                active_pursuers -= 1
                for j in range(num_evaders):
                    assn_model += e[a][j] == 0 
                for c in range(num_pursuers, num_pursuers+num_pursuers_combo):
                    try:
                        p_1, p_2 = [int(o) for o in combo_order["combo"+str(c)+"ev"+str(0)].split(",")]
                    except:
                        print("Error in combo order")
                        print(combo_order)
                        raise SystemExit
                    if p_1 == a or p_2 == a:
                        for j in range(num_evaders):
                            assn_model += e[c][j] == 0 
                
            else:
                # Evader "a-num_pursuers" is Done. No assignements for that evader
                active_evaders -= 1
                for i in range(num_pursuers + num_pursuers_combo):
                    assn_model += e[i][a-num_pursuers] == 0
                    
    print("Active Pursuers:", active_pursuers)
    print("Active Evaders:", active_evaders)
    
    # Constraint: Each evader can be assigned to at most one pursuer or pursuer combination
    for j in range(num_evaders):
        assn_model += xsum(e[i][j] for i in range(num_pursuers + num_pursuers_combo)) <= 1

    # Constraint: All Active Pursuers should have an assignment
    assn_model += xsum(((1 if i < num_pursuers else 2) if active_evaders > active_pursuers else 1)*e[i][j] for i in range(num_pursuers + num_pursuers_combo) for j in range(num_evaders)) == min(active_pursuers, active_evaders)

    # Constraint: If a Combo is assigned, make sure the individual pursuers are not assigned to evaders
    for i in range(num_pursuers, num_pursuers+num_pursuers_combo):
        p_1, p_2 = [int(a) for a in combo_order["combo"+str(i)+"ev"+str(0)].split(",")]
        assn_model += xsum(2*e[i][j] + e[p_1][j] + e[p_2][j] for j in range(num_evaders)) <= 2
        
    # Constraint: Make sure multiple Combinations of same Individual Pursuer is not Assigned
    for i in range(num_pursuers):
        assn_model += xsum(e[d][j] for d in combo_to_pursuer_subsets[i] for j in range(num_evaders)) <= 1

    # # Global constraint to ensure each pursuer is assigned to at most one evader
    # for p in range(num_pursuers):
    #     assn_model += xsum(e[i][j] for j in range(num_evaders) for i in range(num_pursuers + num_pursuers_combo) if p in combo_to_pursuers.get(i, [p])) <= 1

    ###TODO####
    # Check all the constraints

    # Set a maximum gap for optimization (optional)
    assn_model.max_gap = 0.05

    # Solve the problem
    status = assn_model.optimize(max_seconds=300)

    # Print the results
    if status == OptimizationStatus.OPTIMAL:
        print('Optimal solution cost {} found'.format(assn_model.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('Feasible solution cost {} found, best possible: {}'.format(assn_model.objective_value, assn_model.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('No feasible solution found, lower bound is: {}'.format(assn_model.objective_bound))

    # Display the solution
    selected = []
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        print('Solution:')
        
        for j in range(num_evaders):
            selected.append([])
            for i in range(num_pursuers + num_pursuers_combo):
                if e[i][j].x >= 0.9:  # Decision variable value close to 1
                    selected[j].append(i)
        print(selected)

    return selected, combo_order

if __name__ == "__main__":
    print("Preparing for the simulaiton... \n")

    num_pursuer = 3
    num_evader = 1

    env = PursuitEvasion(num_pursuer, num_evader, 1, 0.05)

    T = 30  # total simulation time T = [0.125s (25 A0 by D0), 0.165s (33 A2 by D2), 0.295s (59 A7 by D1),  0.340S (68 A4 by D3),  o.655s (131 A3 by D1), 0.720s (144 A6 by D0), 0.940s (188 A5 by D3), 1.455s (291 A1 by D3)]
    deltat = 0.1 # calculation time interval
    times = int(T/deltat)

    observations, dones = env.reset()


    pursuer_initials = env.pursuers
    evaders_initials = env.evaders

    pursuers_trajectory  = [[] for _ in range(num_pursuer)]
    evaders_trajectory = [[] for _ in range(num_evader)]

    # for plotting
    pursuers_x = [[] for _ in range(num_pursuer)]
    pursuers_y = [[] for _ in range(num_pursuer)]
    evaders_x = [[] for _ in range(num_evader)]
    evaders_y = [[] for _ in range(num_evader)]

    # mip results
    capture_decisions = []

    # load the initial states
    current_pursuers = pursuer_initials
    current_evaders = evaders_initials
    for i in range(num_pursuer):
        pursuers_trajectory[i].append(current_pursuers[i].tolist())
        pursuers_x[i].append(current_pursuers[i][0])
        pursuers_y[i].append(current_pursuers[i][1])

    for j in range(num_evader):
        evaders_trajectory[j].append(current_evaders[j].tolist())
        evaders_x[j].append(current_evaders[j][0])
        evaders_y[j].append(current_evaders[j][1])


    # initialize the captured results
    pursuers_status_logs = []
    pursuers_status = [0 for _ in range(num_pursuer)]
    stops_index = []  # the list stores the indexes of pursuers that has been captured
    pursuers_status_logs.append(pursuers_status)

    car_1v1 = PursuitEvasion1v1(pursuer_R = 0.05, evader_R = 0.05)
    car_2v1 = PursuitEvasion2v1(pursuer_R = 0.05, evader_R = 0.05)

    grid_size  = 40
    g_1v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([grid_size, grid_size, grid_size, grid_size]), [])

    grid_size  = 20

    g_2v1 = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size]), [])


    print("The simulation starts: \n")
    # simulation starts
    for ts in range(0, times):

        actions = [np.zeros((2,))]* (env.num_pursuers + env.num_evaders)

        agent_controls = [(0,0) for _ in range(env.num_evaders + env.num_pursuers)]
        Ic, one_v_one_value_list = capture_1vs1(current_pursuers, current_evaders, stops_index)

        Pc, two_v_one_value_list = capture_2vs1(current_pursuers, current_evaders)

        ### Do Assignment ###
        selected, combo_order = lp_solver(one_v_one_value_list, two_v_one_value_list, num_pursuer, num_evader, dones)
        #   print(f"The MIP result at iteration{ts} is {selected}. \n")
        #   print(f"The combo Order is {combo_order}")
        capture_decisions.append(selected)  # document the capture results

        ### Get Control ###
        # Assign Random controls to pursuers and change it if assigned an evader
        actions[:env.num_pursuers] = [np.random.uniform(-1.0, 1.0, size=(2,))]*env.num_pursuers
        for j in range(len(selected)):
            if len(selected[j]) == 0:
                pass
            elif selected[j][-1]<env.num_pursuers:
                p_sel = selected[j][-1]
                comb_state = (current_pursuers[p_sel][0], current_pursuers[p_sel][1], current_evaders[j][0], current_evaders[j][1])
                pur_action, ev_action = pur_ev_control1v1_1slice(car_1v1, g_1v1, comb_state)
                actions[p_sel] = pur_action
                actions[env.num_pursuers+j] = ev_action
            else:
                p_1 = int(combo_order["combo"+str(selected[j][-1])+"ev"+str(j)][0])
                p_2 = int(combo_order["combo"+str(selected[j][-1])+"ev"+str(j)][1])
                comb_state = (current_pursuers[p_1][0], current_pursuers[p_1][1], current_pursuers[p_2][0], current_pursuers[p_2][1], current_evaders[j][0], current_evaders[j][1])
                
                p1_action, p2_action,ev_action = pur_ev_control2v1_1slice(car_2v1, g_2v1, comb_state)
                actions[p_1] = p1_action
                actions[p_2] = p2_action
                actions[env.num_pursuers+j] = ev_action

        #   print(actions)
        #   raise SystemExit()

        ### Update Positions/use Gym env ###
        _,_,dones,_,_ = env.step(actions)

        if sum(dones)>(num_pursuer+num_evader-1):
            print("Episode Done: ", deltat*(ts+1))
            print(times)
            break
        current_pursuers = env.pursuers
        current_evaders = env.evaders

        for i in range(num_pursuer):
            pursuers_trajectory[i].append(current_pursuers[i].tolist())
            
        for j in range(num_evader):
            evaders_trajectory[j].append(current_evaders[j].tolist())

        print(f"Step Done: {sum(dones)}")

        print("Loop Done. Dumping...")

        with open("pur_pos.txt", "w") as f:
            json.dump(pursuers_trajectory, f)

        with open("ev_pos.txt", "w") as f:
            json.dump(evaders_trajectory, f)

    print("Saved Files")

