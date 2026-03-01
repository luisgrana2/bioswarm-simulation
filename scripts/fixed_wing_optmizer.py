import pathlib
import subprocess
import json
import numpy as np
import pandas as pd
import cma

PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator_fixed"
)

def get_data_from_csv(filename):
    df = pd.read_csv(filename)
    bird_ids = sorted(df['bird_id'].unique())
    num_birds = len(bird_ids)
    
    mean_positions = df.groupby('step')[['px', 'py']].mean().reset_index()
    mean_positions = mean_positions.sort_values('step')
    indices = np.linspace(0, len(mean_positions) - 1, int(len(mean_positions)/10), dtype=int)
    sampled_path = mean_positions.iloc[indices][['px', 'py']].values
    
    first_step = df['step'].min()
    first_frame_data = df[df['step'] == first_step].sort_values('bird_id')
    start_positions = first_frame_data[['px', 'py']].values
    
    bird_trajectories = []
    for b_id in bird_ids:
        b_data = df[df['bird_id'] == b_id].sort_values('step')
        bird_trajectories.append(b_data[['px', 'py']].values)
        
    return sampled_path, num_birds, start_positions, bird_trajectories

def parse_output(data: str):
    trajectories = []
    agents_count = -1
    for idx, line in enumerate(data.splitlines()):
        if idx == 0:
            agents_count = int(line.strip())
            for _ in range(agents_count):
                trajectories.append([])
            continue
        values = list(map(float, line.strip().split()))
        if len(values) < 3 * agents_count + 3:
            continue
        for agent_idx in range(agents_count):
            trajectories[agent_idx].append((values[3 * agent_idx], values[3 * agent_idx + 1], values[3 * agent_idx + 2]))
    return trajectories

def run_simulator(controller_parameters, base_environment):
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters:
        cmd.append(str(val))
    for val in base_environment:
        cmd.append(str(val))
        
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None
    return parse_output(result.stdout.decode())

def calculate_fitness(trajectories, bird_trajectories, waypoints):
    traj_array = np.array(trajectories)
    num_agents = traj_array.shape[0]
    num_steps = traj_array.shape[1]
    
    total_tracking_error = 0.0
    
    for i in range(num_agents):
        sim_path = traj_array[i, :, :2]
        ref_path = bird_trajectories[i]
        
        ref_indices = np.linspace(0, len(ref_path) - 1, num_steps)
        ref_interpolated = np.zeros((num_steps, 2))
        ref_interpolated[:, 0] = np.interp(ref_indices, np.arange(len(ref_path)), ref_path[:, 0])
        ref_interpolated[:, 1] = np.interp(ref_indices, np.arange(len(ref_path)), ref_path[:, 1])
        
        distances = np.linalg.norm(sim_path - ref_interpolated, axis=1)
        total_tracking_error += np.mean(distances)
        
    tracking_cost = total_tracking_error / num_agents
    
    collision_cost = 0.0
    collision_threshold = 1.0 
    
    for step in range(num_steps):
        step_positions = traj_array[:, step, :2]
        diffs = step_positions[:, np.newaxis, :] - step_positions[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf) 
        
        num_collisions = np.sum(dists < collision_threshold) / 2.0
        collision_cost += num_collisions * 1000.0 

    com_last_step = np.mean(traj_array[:, -1, :2], axis=0)
    final_dist_to_last = np.linalg.norm(com_last_step - waypoints[-1])
    
    completion_penalty = 0.0
    if final_dist_to_last > 20.0:
        completion_penalty = 5000.0 + final_dist_to_last * 100.0
        
    time_cost = num_steps * 0.5
    
    total_cost = time_cost + (tracking_cost * 10.0) + completion_penalty + collision_cost
    return total_cost

def objective_function(params, mode, clips_data):
    max_speed, rep_radius, rep_weight, param_a, param_b, param_c = params
    
    if max_speed < 3.0 or rep_radius < 1.0 or rep_weight < 0.0 or param_a < 0.0 or param_b < 0.0 or param_c < 0.0:
        return 100000.0
        
    controller_weights = [mode, max_speed, rep_radius, rep_weight, param_a, param_b, param_c]
    
    total_fitness = 0.0
    
    for clip_id, offset in clips_data.items():
        offset_x, offset_y = offset
        
        csv_file = f'../config/reference_trajectory_{clip_id}.csv'
        waypoints, drone_count, start_positions, bird_trajectories = get_data_from_csv(csv_file)
        
        first_goal = waypoints[0]
        
        shifted_positions = start_positions.copy()
        shifted_positions[:, 0] += offset_x
        shifted_positions[:, 1] += offset_y
        
        distances_to_goal = [np.linalg.norm(p - first_goal[:2]) for p in shifted_positions]
        closest_idx = np.argmin(distances_to_goal)
        
        start_positions[[0, closest_idx]] = start_positions[[closest_idx, 0]]
        bird_trajectories[0], bird_trajectories[closest_idx] = bird_trajectories[closest_idx], bird_trajectories[0]
        
        flat_starts = []
        for p in start_positions:
            start_x = p[0] - offset_x
            start_y = p[1] - offset_y
            angle_to_goal = np.arctan2(first_goal[1] - start_y, first_goal[0] - start_x)
            flat_starts.extend([start_x, start_y, 10.0, angle_to_goal])
            
        flat_waypoints = []
        for wp in waypoints:
            flat_waypoints.extend([wp[0], wp[1], 10.0])
            
        base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints
        
        trajectories = run_simulator(controller_weights, base_environment)
        
        if trajectories is None or len(trajectories) == 0 or len(trajectories[0]) == 0:
            return 100000.0 
            
        fitness = calculate_fitness(trajectories, bird_trajectories, waypoints)
        total_fitness += fitness
        
    return total_fitness / len(clips_data)

if __name__ == "__main__":
    
    clip_offsets = {
        "006": (0.0, -50.0),
        "011": (0.0, 50.0),
        "015": (0.0, -50.0),
        "021": (-25.0, -50.0)
    }
    
    modes = [0, 1, 2]
    mode_names = ["Consensus Leaderless", "Fixed Leader", "Dynamic Leader"]
    
    all_best_params = []
    all_best_errors = []
    
    initial_params = [6.0, 12.0, 4.0, 0.8, 1.2, 1.5]
    initial_std = 1.5
    
    for target_mode, mode_name in zip(modes, mode_names):
        print(f"\nStarting CMA-ES optimization for Option {target_mode + 1} ({mode_name})")
        
        es = cma.CMAEvolutionStrategy(initial_params, initial_std)
        generation = 0
        
        while not es.stop():
            solutions = es.ask()
            fitness_values = [objective_function(sol, target_mode, clip_offsets) for sol in solutions]
            es.tell(solutions, fitness_values)
            es.disp()
            
            if generation > 0 and generation % 20 == 0:
                current_best = es.result.xbest
                print("Current best parameters at generation", generation)
                print("max_speed:", current_best[0])
                print("rep_radius:", current_best[1])
                print("rep_weight:", current_best[2])
                print("param_a:", current_best[3])
                print("param_b:", current_best[4])
                print("param_c:", current_best[5])
                
            generation += 1
            
        print(f"Optimization finished for Option {target_mode + 1}.")
        all_best_params.append(es.result.xbest)
        all_best_errors.append(es.result.fbest)
        
    print("\nOptimization complete for all options. Final Best Parameters:")
    
    output_path = pathlib.Path(__file__).parent / ".." / "config" / "optimized_parameters.json"
    results_dict = {}
    
    if output_path.exists():
        with open(output_path, "r") as json_file:
            try:
                results_dict = json.load(json_file)
            except json.JSONDecodeError:
                pass
    
    for target_mode, mode_name, best_params, best_error in zip(modes, mode_names, all_best_params, all_best_errors):
        print(f"\nOption {target_mode + 1} - {mode_name}:")
        print("max_speed:", best_params[0])
        print("rep_radius:", best_params[1])
        print("rep_weight:", best_params[2])
        print("param_a:", best_params[3])
        print("param_b:", best_params[4])
        print("param_c:", best_params[5])
        print("final_error:", best_error)
        
        dict_key = f"Option_{target_mode + 1}_{mode_name.replace(' ', '_')}"
        results_dict[dict_key] = {
            "max_speed": float(best_params[0]),
            "rep_radius": float(best_params[1]),
            "rep_weight": float(best_params[2]),
            "param_a": float(best_params[3]),
            "param_b": float(best_params[4]),
            "param_c": float(best_params[5]),
            "final_error": float(best_error)
        }
        
    with open(output_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
        
    print(f"\nAll final parameters and errors have been successfully saved to {output_path}")