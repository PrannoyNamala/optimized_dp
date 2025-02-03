import os
import yaml
from collections import defaultdict
import numpy as np

# Path to the base directory
base_dir = "experiment_log"

# Dictionary to store cumulative values and counts
results = defaultdict(lambda: {
    "ev_captures": 0, 
    "timesteps": 0, 
    "count": 0, 
    "max_timesteps": float('-inf'), 
    "min_timesteps": float('inf'), 
    "max_captures": float('-inf'), 
    "min_captures": float('inf'), 
    "captures_list": [], 
    "reason_counts": defaultdict(int)
})

# Traverse the directory structure
for main_folder in os.listdir(base_dir):
    main_folder_path = os.path.join(base_dir, main_folder)

    if os.path.isdir(main_folder_path):
        for subfolder in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder)

            if os.path.isdir(subfolder_path):
                metrics_file = os.path.join(subfolder_path, "metrics.yaml")

                if os.path.isfile(metrics_file):
                    with open(metrics_file, "r") as file:
                        metrics = yaml.safe_load(file)

                        # Extract values and update results
                        ev_captures = metrics.get("ev_captures", 0)
                        timesteps = metrics.get("timesteps", 0)
                        reason = metrics.get("Reason", "Unknown")

                        results[main_folder]["ev_captures"] += ev_captures
                        results[main_folder]["timesteps"] += timesteps
                        results[main_folder]["count"] += 1

                        # Update max and min timesteps
                        if timesteps > results[main_folder]["max_timesteps"]:
                            results[main_folder]["max_timesteps"] = timesteps
                        if timesteps < results[main_folder]["min_timesteps"]:
                            results[main_folder]["min_timesteps"] = timesteps

                        # Update max and min captures
                        if ev_captures > results[main_folder]["max_captures"]:
                            results[main_folder]["max_captures"] = ev_captures
                        if ev_captures < results[main_folder]["min_captures"]:
                            results[main_folder]["min_captures"] = ev_captures

                        # Append captures to the list for std deviation
                        results[main_folder]["captures_list"].append(ev_captures)

                        # Count reasons
                        results[main_folder]["reason_counts"][reason] += 1

# Calculate averages, std deviation, and print results
for main_folder, data in results.items():
    if data["count"] > 0:
        avg_ev_captures = data["ev_captures"] / data["count"]
        avg_timesteps = data["timesteps"] / data["count"]
        captures_std_dev = np.std(data["captures_list"])

        print(f"{main_folder}:")
        print(f"  Average ev_captures: {avg_ev_captures:.2f}")
        print(f"  Average timesteps: {avg_timesteps:.2f}")
        print(f"  Max timesteps: {data['max_timesteps']}")
        print(f"  Min timesteps: {data['min_timesteps']}")
        print(f"  Max captures: {data['max_captures']}")
        print(f"  Min captures: {data['min_captures']}")
        print(f"  Std deviation of captures: {captures_std_dev:.2f}")
        print(f"  Reason counts:")
        for reason, count in data["reason_counts"].items():
            print(f"    {reason}: {count}")
    else:
        print(f"{main_folder} has no valid metrics.yaml files.")
