import pathlib
import subprocess
import json
import numpy as np
import pandas as pd
import cma
import concurrent.futures

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
        
        # Ensure the output contains exactly the expected coordinates per agent
        if len(values) != 3 * agents_count:
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
        
    try:
        # Enforce a strict timeout to prevent any infinite loops in the C++ backend
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0:
            return None
        return parse_output(result.stdout.decode())
    except subprocess.TimeoutExpired:
        return None

def calculate_fitness(trajectories, bird_trajectories, offset_x, offset_y):
    traj_array = np.array(trajectories)
    num_agents = traj_array.shape[0]
    num_steps = traj_array.shape[1]
    
    all_distances = []
    
    for i in range(num_agents):
        # Shift the simulated coordinates back to the original global frame
        sim_path = traj_array[i, :, :2].copy()
        sim_path[:, 0] += offset_x
        sim_path[:, 1] += offset_y
        
        ref_path = bird_trajectories[i]
        
        # Interpolate the paths so the evaluation compares matched time steps
        ref_indices = np.linspace(0, 1, len(ref_path))
        sim_indices = np.linspace(0, 1, num_steps)
        
        sim_interp_x = np.interp(ref_indices, sim_indices, sim_path[:, 0])
        sim_interp_y = np.interp(ref_indices, sim_indices, sim_path[:, 1])
        sim_interp = np.column_stack((sim_interp_x, sim_interp_y))
        
        # Accumulate the distance error in meters
        distances = np.linalg.norm(sim_interp - ref_path, axis=1)
        all_distances.extend(distances)
        
    # Evaluate the overall fitness using the median to discard outliers naturally
    tracking_error_meters = np.median(all_distances)
    
    # Introduce a soft penalty for agents overlapping their physical space
    collision_cost = 0.0
    collision_threshold = 1.0 
    
    for step in range(num_steps):
        step_positions = traj_array[:, step, :2]
        diffs = step_positions[:, np.newaxis, :] - step_positions[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf) 
        num_collisions = np.sum(dists < collision_threshold) / 2.0
        collision_cost += num_collisions * 0.1 

    # Combine the median path deviation with the spatial collision penalty
    total_cost = tracking_error_meters + collision_cost
    return total_cost

# Define broader search space boundaries to explore extreme parameter combinations
lower_bounds = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0]
upper_bounds = [50.0, 100.0, 50.0, 20.0, 20.0, 20.0]

def objective_function(params, mode, clips_data):
    # Clip the parameters so the optimizer guesses remain within realistic bounds
    params = np.clip(params, lower_bounds, upper_bounds)
    max_speed, rep_radius, rep_weight, param_a, param_b, param_c = params
    
    controller_weights = [mode, max_speed, rep_radius, rep_weight, param_a, param_b, param_c]
    
    total_fitness = 0.0
    
    for clip_id, offset in clips_data.items():
        offset_x, offset_y = offset
        
        csv_file = f'../config/reference_trajectory_{clip_id}.csv'
        waypoints, drone_count, start_positions, bird_trajectories = get_data_from_csv(csv_file)
        
        flat_starts = []
        for p in start_positions:
            start_x = p[0] - offset_x
            start_y = p[1] - offset_y
            
            # Send only positional coordinates since the backend determines the starting heading
            flat_starts.extend([start_x, start_y, 10.0])
            
        flat_waypoints = []
        for wp in waypoints:
            flat_waypoints.extend([wp[0], wp[1], 10.0])
            
        base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints
        
        trajectories = run_simulator(controller_weights, base_environment)
        
        if trajectories is None or len(trajectories) == 0 or len(trajectories[0]) == 0:
            # Return a significant but finite penalty value to preserve the optimization gradient
            return 500.0 
            
        fitness = calculate_fitness(trajectories, bird_trajectories, offset_x, offset_y)
        total_fitness += fitness
        
    return total_fitness / len(clips_data)

# Dispatch evaluations across multiple CPU cores
def evaluate_population(solutions, target_mode, clip_offsets):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(objective_function, sol, target_mode, clip_offsets) for sol in solutions]
        return [f.result() for f in futures]

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
    initial_std = 3.0
    
    for target_mode, mode_name in zip(modes, mode_names):
        print(f"\nStarting CMA-ES optimization for Option {target_mode + 1} ({mode_name})")
        
        es = cma.CMAEvolutionStrategy(initial_params, initial_std, {'bounds': [lower_bounds, upper_bounds]})
        generation = 0
        
        while not es.stop():
            solutions = es.ask()
            
            # Execute the evaluations in parallel to speed up the process
            fitness_values = evaluate_population(solutions, target_mode, clip_offsets)
            
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
    
    output_path = pathlib.Path(__file__).parent / ".." / "config" / "optimized_parameters_wrt_real.json"
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