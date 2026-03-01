import pathlib
import subprocess
import numpy as np
import pandas as pd

PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator_fixed"
)

def get_data_from_csv(filename):
    df = pd.read_csv(filename)
    num_birds = len(df['bird_id'].unique())
    
    mean_positions = df.groupby('step')[['px', 'py']].mean().reset_index()
    mean_positions = mean_positions.sort_values('step')
    indices = np.linspace(0, len(mean_positions) - 1, 15, dtype=int)
    sampled_path = mean_positions.iloc[indices][['px', 'py']].values
    
    first_step = df['step'].min()
    first_frame_data = df[df['step'] == first_step].sort_values('bird_id')
    start_positions = first_frame_data[['px', 'py']].values
    
    return sampled_path, num_birds, start_positions

def parse_output(data: str) -> list[list[tuple[float, float, float]]]:
    trajectories = []
    agents_count = -1
    for i, line in enumerate(data.splitlines()):
        if i == 0:
            agents_count = int(line.strip())
            for _ in range(agents_count):
                trajectories.append([])
            continue
        values = list(map(float, line.strip().split()))
        if len(values) != 3 * agents_count:
            continue
        for j in range(agents_count):
            trajectories[j].append((values[3 * j], values[3 * j + 1], values[3 * j + 2]))
    return trajectories

def run_simulator(controller_parameters: list[float]) -> list[list[tuple[float, float, float]]]:
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters:
        cmd.append(str(val))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return []
    return parse_output(result.stdout.decode())

def evaluate_parameters(x, mode, environments):
    rep_rad, rep_weight, param_a, param_b, param_c = x
    
    if rep_rad <= 0 or rep_weight < 0 or param_a < 0 or param_b < 0 or param_c < 0:
        return 1e6 
        
    total_cost = 0.0
    
    for base_env in environments:
        full_params = [mode, 6.0, rep_rad, rep_weight, param_a, param_b, param_c] + base_env
        trajectories = run_simulator(full_params)
        
        if not trajectories or len(trajectories[0]) == 0:
            return 1e6
            
        frames_taken = len(trajectories[0])
        final_positions = np.array([agent_traj[-1] for agent_traj in trajectories])
        target = np.array([base_env[-3], base_env[-2], base_env[-1]])
        
        distances_to_target = np.linalg.norm(final_positions - target, axis=1)
        mean_distance = np.mean(distances_to_target)
        
        cost = frames_taken + (mean_distance * 100.0)
        total_cost += cost
        
    return total_cost / len(environments)