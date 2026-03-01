import pathlib
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

clip = "011"
PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator"
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
        raise RuntimeError(result.stderr.decode())
    return parse_output(result.stdout.decode())

def animate_trajectories(res_cons, res_fixed, res_dyn, waypoints):
    max_frames = max(len(res_cons[0]), len(res_fixed[0]), len(res_dyn[0]))
    
    for res in [res_cons, res_fixed, res_dyn]:
        for agent_traj in res:
            while len(agent_traj) < max_frames:
                agent_traj.append(agent_traj[-1])
                
    num_agents = len(res_cons)
    fig, (ax_cons, ax_fixed, ax_dyn) = plt.subplots(1, 3, figsize=(20, 6))
    
    scat_cons = ax_cons.scatter([], [], c='blue', s=40)
    scat_fixed = ax_fixed.scatter([], [], c='blue', s=40)
    leader_fixed = ax_fixed.scatter([], [], c='green', s=100, zorder=5)
    scat_dyn = ax_dyn.scatter([], [], c='blue', s=40)
    leader_dyn = ax_dyn.scatter([], [], c='green', s=100, zorder=5)
    
    target_scat_cons = ax_cons.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_fixed = ax_fixed.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_dyn = ax_dyn.scatter([], [], c='red', s=120, marker='X', zorder=6)
    
    line_dyn, = ax_dyn.plot([], [], 'r--', alpha=0.5)
    
    min_x = np.min(waypoints[:, 0]) - 15.0
    max_x = np.max(waypoints[:, 0]) + 15.0
    min_y = np.min(waypoints[:, 1]) - 15.0
    max_y = np.max(waypoints[:, 1]) + 15.0
    
    for ax, title in zip([ax_cons, ax_fixed, ax_dyn], ['Consensus', 'Fixed Leader', 'Dynamic Leader']):
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box') 
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='gray', s=60, marker='o', alpha=0.5)
        
    target_idx_c = 0
    target_idx_f = 0
    target_idx_d = 0
    target_threshold = 4.0
    active_dyn_leader = 0
        
    def update(frame):
        nonlocal target_idx_c, target_idx_f, target_idx_d, active_dyn_leader
        
        pos_c = [res_cons[i][frame][:2] for i in range(num_agents)]
        scat_cons.set_offsets(pos_c)
        com_c = np.mean(pos_c, axis=0)
        if target_idx_c < len(waypoints) - 1 and np.linalg.norm(com_c - waypoints[target_idx_c]) < target_threshold:
            target_idx_c += 1
        target_scat_cons.set_offsets([waypoints[target_idx_c]])
        
        pos_f = [res_fixed[i][frame][:2] for i in range(num_agents)]
        scat_fixed.set_offsets(pos_f)
        leader_fixed.set_offsets(pos_f[0])
        if target_idx_f < len(waypoints) - 1 and np.linalg.norm(pos_f[0] - waypoints[target_idx_f]) < target_threshold:
            target_idx_f += 1
        target_scat_fixed.set_offsets([waypoints[target_idx_f]])
        
        pos_d = [res_dyn[i][frame][:2] for i in range(num_agents)]
        
        min_dist_to_current = min([np.linalg.norm(np.array(p) - waypoints[target_idx_d]) for p in pos_d])
        if target_idx_d < len(waypoints) - 1 and min_dist_to_current < target_threshold:
            target_idx_d += 1
            
        target_scat_dyn.set_offsets([waypoints[target_idx_d]])
        
        dists_to_current = [np.linalg.norm(np.array(p) - waypoints[target_idx_d]) for p in pos_d]
        candidate_idx = np.argmin(dists_to_current)
        
        if (dists_to_current[active_dyn_leader] - dists_to_current[candidate_idx]) > 2.0:
            active_dyn_leader = candidate_idx
            
        scat_dyn.set_offsets(pos_d)
        leader_dyn.set_offsets(pos_d[active_dyn_leader])
        
        leader_pos = pos_d[active_dyn_leader]
        target_pos = waypoints[target_idx_d]
        line_dyn.set_data([leader_pos[0], target_pos[0]], [leader_pos[1], target_pos[1]])
        
        return scat_cons, target_scat_cons, scat_fixed, leader_fixed, target_scat_fixed, scat_dyn, leader_dyn, target_scat_dyn, line_dyn
        
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=True)
    ani.save(f'swarm_waypoints_{clip}.mp4', writer='ffmpeg', fps=10)
    print("Animation saved successfully.")

if __name__ == "__main__":
    csv_path = f'../config/reference_trajectory_{clip}.csv'
    waypoints, drone_count, start_positions = get_data_from_csv(csv_path)
    
    flat_starts = []
    for p in start_positions:
        flat_starts.extend([p[0], p[1], 10.0])
        
    flat_waypoints = []
    for wp in waypoints:
        flat_waypoints.extend([wp[0], wp[1], 10.0])
        
    base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints

    # Default parameters before optimization
    max_speed = 6.0
    repulsion_radius = 12.0
    repulsion_weight = 4.0
    param_a = 0.8
    param_b = 1.2
    param_c = 1.5
    
    # You can replace the values above with the results from optimize_cma.py
    controller_weights = [max_speed, repulsion_radius, repulsion_weight, param_a, param_b, param_c]

    print(f"Starting simulation with {drone_count} UAVs mapped to exact starting positions...")

    print("Running Consensus Mode...")
    res_cons = run_simulator([0] + controller_weights + base_environment)
    
    print("Running Fixed Leader Mode...")
    res_fixed = run_simulator([1] + controller_weights + base_environment)
    
    print("Running Dynamic Leader Mode...")
    res_dyn = run_simulator([2] + controller_weights + base_environment)
    
    print("Generating animation...")
    animate_trajectories(res_cons, res_fixed, res_dyn, waypoints)