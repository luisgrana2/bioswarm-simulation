import pathlib
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator_fixed"
)

def parse_output(data: str):
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

def run_simulator(controller_parameters, base_environment):
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters + base_environment:
        cmd.append(str(val))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return parse_output(result.stdout.decode())

waypoints = np.array([
    [20.0, 20.0],
    [-20.0, 30.0],
    [-20.0, -10.0],
    [30.0, -20.0],
    [0.0, 0.0]
])

drone_count = 1
flat_starts = [0.0, 0.0, 10.0]
flat_waypoints = []
for wp in waypoints:
    flat_waypoints.extend([wp[0], wp[1], 10.0])

base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints
controller_weights = [0, 6.0, 12.0, 4.0, 0.8, 1.2, 1.5]

print("Running executable...")
res = run_simulator(controller_weights, base_environment)
agent_traj = np.array(res[0])

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.set_title('Executable Output: Fixed-Wing Kinematics')
ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', marker='x', s=100, label='Waypoints')

quiver = ax.quiver(agent_traj[0, 0], agent_traj[0, 1], 1, 0, color='blue', scale=15, width=0.015)
trajectory_line, = ax.plot([], [], 'b--', alpha=0.5)

def update(frame):
    x = agent_traj[frame, 0]
    y = agent_traj[frame, 1]
    
    if frame > 0:
        prev_x = agent_traj[frame-1, 0]
        prev_y = agent_traj[frame-1, 1]
        dx = x - prev_x
        dy = y - prev_y
        if np.hypot(dx, dy) > 0.001:
            u, v = dx, dy
        else:
            u, v = quiver.U[0], quiver.V[0]
    else:
        u, v = 1.0, 0.0
        
    norm = np.hypot(u, v)
    if norm > 0:
        u, v = u/norm, v/norm
        
    quiver.set_offsets(np.c_[x, y])
    quiver.set_UVC(u, v)
    trajectory_line.set_data(agent_traj[:frame+1, 0], agent_traj[:frame+1, 1])
    
    return quiver, trajectory_line

max_frames = len(agent_traj)
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=True)
plt.legend()
plt.show()