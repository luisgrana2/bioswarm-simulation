import pathlib
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

clip = "021"

clip_offsets = {
    "006": (0.0, -50.0),
    "011": (0.0, 50.0),
    "015": (0.0, -50.0),
    "021": (-25.0, -50.0)
}

# Check if the clip exists in the dictionary and unpack the tuple
if clip in clip_offsets:
    OFFSET_X, OFFSET_Y = clip_offsets[clip]
    
PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator_fixed"
)

def get_data_from_csv(filename):
    df = pd.read_csv(filename)
    bird_ids = sorted(df['bird_id'].unique())
    num_birds = len(bird_ids)
    
    mean_positions = df.groupby('step')[['px', 'py']].mean().reset_index()
    mean_positions = mean_positions.sort_values('step')
    indices = np.linspace(0, len(mean_positions) - 1, 15, dtype=int)
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
    target_trajectory = []
    agents_count = -1
    for i, line in enumerate(data.splitlines()):
        if i == 0:
            agents_count = int(line.strip())
            for _ in range(agents_count):
                trajectories.append([])
            continue
        values = list(map(float, line.strip().split()))
        if len(values) < 3 * agents_count + 3:
            continue
        for j in range(agents_count):
            trajectories[j].append((values[3 * j], values[3 * j + 1], values[3 * j + 2]))
        target_trajectory.append((values[-3], values[-2], values[-1]))
    return trajectories, target_trajectory

def run_simulator(controller_parameters: list[float]):
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters:
        cmd.append(str(val))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return parse_output(result.stdout.decode())

def animate_trajectories(res_cons, target_cons, res_fixed, target_fixed, res_dyn, target_dyn, waypoints):
    max_frames = max(len(res_cons[0]), len(res_fixed[0]), len(res_dyn[0]))
    
    for res in [res_cons, res_fixed, res_dyn]:
        for agent_traj in res:
            while len(agent_traj) < max_frames:
                agent_traj.append(agent_traj[-1])
                
    while len(target_cons) < max_frames: target_cons.append(target_cons[-1])
    while len(target_fixed) < max_frames: target_fixed.append(target_fixed[-1])
    while len(target_dyn) < max_frames: target_dyn.append(target_dyn[-1])
                
    num_agents = len(res_cons)
    fig, (ax_cons, ax_fixed, ax_dyn) = plt.subplots(1, 3, figsize=(20, 6))
    
    scat_cons = ax_cons.scatter([], [], c='blue', s=20, zorder=4)
    scat_fixed = ax_fixed.scatter([], [], c='blue', s=20, zorder=4)
    leader_fixed = ax_fixed.scatter([], [], c='green', s=80, zorder=5)
    scat_dyn = ax_dyn.scatter([], [], c='blue', s=20, zorder=4)
    leader_dyn = ax_dyn.scatter([], [], c='green', s=80, zorder=5)
    
    target_scat_cons = ax_cons.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_fixed = ax_fixed.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_dyn = ax_dyn.scatter([], [], c='red', s=120, marker='X', zorder=6)
    
    line_dyn, = ax_dyn.plot([], [], 'r--', alpha=0.5)

    init_zeros = np.zeros(num_agents)
    quiver_c = ax_cons.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)
    quiver_f = ax_fixed.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)
    quiver_d = ax_dyn.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)

    fov_angle = 316.4
    half_fov = fov_angle / 2.0
    fov_radius = 6.0
    
    wedges_c = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]
    wedges_f = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]
    wedges_d = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]

    for w in wedges_c: ax_cons.add_patch(w)
    for w in wedges_f: ax_fixed.add_patch(w)
    for w in wedges_d: ax_dyn.add_patch(w)
    
    min_x = np.min(waypoints[:, 0]) - 15.0
    max_x = np.max(waypoints[:, 0]) + 15.0
    min_y = np.min(waypoints[:, 1]) - 15.0
    max_y = np.max(waypoints[:, 1]) + 15.0
    
    for ax, title in zip([ax_cons, ax_fixed, ax_dyn], ['Consensus', 'Fixed Leader', 'Dynamic Leader']):
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y-30, max_y)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box') 
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='gray', s=60, marker='o', alpha=0.5, zorder=2)
        
    first_goal = waypoints[0]
    initial_headings = []
    for i in range(num_agents):
        start_pos_x = res_cons[i][0][0]
        start_pos_y = res_cons[i][0][1]
        angle_to_goal = np.arctan2(first_goal[1] - start_pos_y, first_goal[0] - start_pos_x)
        initial_headings.append(angle_to_goal)

    headings_c = np.array(initial_headings)
    headings_f = np.array(initial_headings)
    headings_d = np.array(initial_headings)
    
    active_dyn_leader = 0

    def update_headings(pos, prev_pos, headings):
        for i in range(len(pos)):
            dx = pos[i][0] - prev_pos[i][0]
            dy = pos[i][1] - prev_pos[i][1]
            if np.hypot(dx, dy) > 0.05:
                headings[i] = np.arctan2(dy, dx)
        return headings

    def update_wedges(wedges, positions, headings):
        for i, w in enumerate(wedges):
            w.set_center(positions[i])
            deg = np.degrees(headings[i])
            w.set_theta1(deg - half_fov)
            w.set_theta2(deg + half_fov)
        
    def update(frame):
        nonlocal active_dyn_leader
        nonlocal headings_c, headings_f, headings_d
        
        pos_c = np.array([res_cons[i][frame][:2] for i in range(num_agents)])
        prev_pos_c = np.array([res_cons[i][max(0, frame-1)][:2] for i in range(num_agents)])
        headings_c = update_headings(pos_c, prev_pos_c, headings_c)
        
        scat_cons.set_offsets(pos_c)
        quiver_c.set_offsets(pos_c)
        quiver_c.set_UVC(np.cos(headings_c), np.sin(headings_c))
        update_wedges(wedges_c, pos_c, headings_c)

        target_scat_cons.set_offsets([target_cons[frame][:2]])
        
        pos_f = np.array([res_fixed[i][frame][:2] for i in range(num_agents)])
        prev_pos_f = np.array([res_fixed[i][max(0, frame-1)][:2] for i in range(num_agents)])
        headings_f = update_headings(pos_f, prev_pos_f, headings_f)

        scat_fixed.set_offsets(pos_f)
        leader_fixed.set_offsets(pos_f[0])
        quiver_f.set_offsets(pos_f)
        quiver_f.set_UVC(np.cos(headings_f), np.sin(headings_f))
        update_wedges(wedges_f, pos_f, headings_f)
        
        wedges_f[0].set_color('green')
        wedges_f[0].set_alpha(0.2)

        target_scat_fixed.set_offsets([target_fixed[frame][:2]])
        
        pos_d = np.array([res_dyn[i][frame][:2] for i in range(num_agents)])
        prev_pos_d = np.array([res_dyn[i][max(0, frame-1)][:2] for i in range(num_agents)])
        headings_d = update_headings(pos_d, prev_pos_d, headings_d)
        
        target_scat_dyn.set_offsets([target_dyn[frame][:2]])
        
        dists_to_current = [np.linalg.norm(np.array(p) - target_dyn[frame][:2]) for p in pos_d]
        candidate_idx = np.argmin(dists_to_current)
        
        if (dists_to_current[active_dyn_leader] - dists_to_current[candidate_idx]) > 2.0:
            wedges_d[active_dyn_leader].set_color('blue')
            wedges_d[active_dyn_leader].set_alpha(0.1)
            active_dyn_leader = candidate_idx
            
        scat_dyn.set_offsets(pos_d)
        leader_dyn.set_offsets(pos_d[active_dyn_leader])
        quiver_d.set_offsets(pos_d)
        quiver_d.set_UVC(np.cos(headings_d), np.sin(headings_d))
        update_wedges(wedges_d, pos_d, headings_d)
        
        wedges_d[active_dyn_leader].set_color('green')
        wedges_d[active_dyn_leader].set_alpha(0.2)
        
        leader_pos = pos_d[active_dyn_leader]
        line_dyn.set_data([leader_pos[0], target_dyn[frame][0]], [leader_pos[1], target_dyn[frame][1]])
        
        return (scat_cons, target_scat_cons, scat_fixed, leader_fixed, target_scat_fixed, 
                scat_dyn, leader_dyn, target_scat_dyn, line_dyn, 
                quiver_c, quiver_f, quiver_d) + tuple(wedges_c) + tuple(wedges_f) + tuple(wedges_d)
        
    skip_step = 5
    animation_frames = list(range(0, max_frames, skip_step))
    
    standard_fps = 10.0 * (5.0 / skip_step)
    video_duration = len(animation_frames) / standard_fps
    
    if video_duration < 5.0:
        adjusted_fps = len(animation_frames) / 5.0
    else:
        adjusted_fps = standard_fps
        
    frame_interval = int(1000 / adjusted_fps) if adjusted_fps > 0 else 100
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, interval=frame_interval, blit=True)
    
    ani.save(f'swarm_waypoints_{clip}_fixed.mp4', writer='ffmpeg', fps=adjusted_fps)
    print("Animation saved successfully.")

def print_trajectory_metrics(mode_name, trajectories, bird_trajectories):
    traj_array = np.array(trajectories)
    num_agents = traj_array.shape[0]
    num_steps = traj_array.shape[1]
    
    total_error = 0.0
    
    for i in range(num_agents):
        sim_path = traj_array[i, :, :2]
        ref_path = bird_trajectories[i]
        
        ref_indices = np.linspace(0, len(ref_path) - 1, num_steps)
        ref_interpolated = np.zeros((num_steps, 2))
        ref_interpolated[:, 0] = np.interp(ref_indices, np.arange(len(ref_path)), ref_path[:, 0])
        ref_interpolated[:, 1] = np.interp(ref_indices, np.arange(len(ref_path)), ref_path[:, 1])
        
        distances = np.linalg.norm(sim_path - ref_interpolated, axis=1)
        total_error += np.mean(distances)
        
    global_mean = total_error / num_agents
    
    print(f"--- {mode_name} Metrics ---")
    print(f"Maximum trajectory time: {num_steps} frames")
    print(f"Average individual distance to true bird trajectory: {global_mean:.2f} meters\n")

if __name__ == "__main__":
    csv_path = f'../config/reference_trajectory_{clip}.csv'
    waypoints, drone_count, start_positions, bird_trajectories = get_data_from_csv(csv_path)
    
    first_goal = waypoints[0]
    
    # Apply the exact same shift before calculating the closest agent
    shifted_positions = start_positions.copy()
    shifted_positions[:, 0] += OFFSET_X
    shifted_positions[:, 1] += OFFSET_Y
    
    distances_to_goal = [np.linalg.norm(p - first_goal[:2]) for p in shifted_positions]
    closest_idx = np.argmin(distances_to_goal)
    
    # Swap the closest agent into the 0 index so it becomes the fixed leader
    start_positions[[0, closest_idx]] = start_positions[[closest_idx, 0]]
    
    # Swap the true trajectory data to keep the final error metrics aligned
    bird_trajectories[0], bird_trajectories[closest_idx] = bird_trajectories[closest_idx], bird_trajectories[0]
    
    flat_starts = []
    
    for p in start_positions:
        start_x = p[0] + OFFSET_X
        start_y = p[1] + OFFSET_Y 
        angle_to_goal = np.arctan2(first_goal[1] - start_y, first_goal[0] - start_x)
        flat_starts.extend([start_x, start_y, 10.0, angle_to_goal])
        
    flat_waypoints = []
    for wp in waypoints:
        flat_waypoints.extend([wp[0], wp[1], 10.0])
        
    base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints

    default_weights = [6.5021, 9.2958, 5.9113, 1.0114, 3.4482, 3.1759]
    weights_cons = default_weights.copy()
    weights_fixed = default_weights.copy()
    weights_dyn = default_weights.copy()

    json_path = pathlib.Path(__file__).parent / ".." / "config" / "optimized_parameters.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                saved_params = json.load(f)
                
            def extract_params(mode_key, fallback):
                if mode_key in saved_params:
                    d = saved_params[mode_key]
                    return [
                        d.get("max_speed", fallback[0]),
                        d.get("rep_radius", fallback[1]),
                        d.get("rep_weight", fallback[2]),
                        d.get("param_a", fallback[3]),
                        d.get("param_b", fallback[4]),
                        d.get("param_c", fallback[5])
                    ]
                return fallback

            weights_cons = extract_params("Option_1_Consensus_Leaderless", default_weights)
            weights_fixed = extract_params("Option_2_Fixed_Leader", default_weights)
            weights_dyn = extract_params("Option_3_Dynamic_Leader", default_weights)
            print("Successfully loaded optimal parameters from JSON.\n")
            
        except Exception as e:
            print(f"Could not load JSON, using default weights. Error: {e}\n")
    else:
        print("JSON file not found, using default weights.\n")

    print(f"Starting kinematic simulation with {drone_count} agents mapped to exact starting positions...\n")

    print("Running Consensus Mode...")
    res_cons, target_cons = run_simulator([0] + weights_cons + base_environment)
    print_trajectory_metrics("Consensus Leaderless", res_cons, bird_trajectories)
    
    print("Running Fixed Leader Mode...")
    res_fixed, target_fixed = run_simulator([1] + weights_fixed + base_environment)
    print_trajectory_metrics("Fixed Leader", res_fixed, bird_trajectories)
    
    print("Running Dynamic Leader Mode...")
    res_dyn, target_dyn = run_simulator([2] + weights_dyn + base_environment)
    print_trajectory_metrics("Dynamic Leader", res_dyn, bird_trajectories)
    
    print("Generating animation...")
    animate_trajectories(res_cons, target_cons, res_fixed, target_fixed, res_dyn, target_dyn, waypoints)