import cma
from opt_utils import get_data_from_csv, evaluate_parameters

# Optimal Parameters: Repulsion Radius=10.75, Repulsion Weight=3.19, Param A=10.87, Param B=12.08, Param C=2.76
if __name__ == "__main__":
    clips = ["006", "011", "015", "021"] 
    all_environments = []
    
    for clip in clips:
        csv_path = f'../config/reference_trajectory_{clip}.csv'
        waypoints, drone_count, start_positions = get_data_from_csv(csv_path)
        
        flat_starts = []
        for p in start_positions:
            flat_starts.extend([p[0], p[1], 10.0])
            
        flat_waypoints = []
        for wp in waypoints:
            flat_waypoints.extend([wp[0], wp[1], 10.0])
            
        base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints
        all_environments.append(base_environment)
        
    target_mode = 1
    print(f"\nStarting CMA-ES Optimization for FIXED LEADER Mode...")
    
    initial_guess = [12.0, 4.0, 0.8, 1.2, 1.5]
    initial_std = 1.0
    
    optimizer = cma.CMAEvolutionStrategy(initial_guess, initial_std, {'bounds': [0, None], 'popsize': 10, 'maxiter': 30})
    
    while not optimizer.stop():
        solutions = optimizer.ask()
        fitness_values = [evaluate_parameters(x, target_mode, all_environments) for x in solutions]
        optimizer.tell(solutions, fitness_values)
        optimizer.disp()
        
    best_parameters = optimizer.result.xbest
    print(f"\nOptimization Complete for FIXED LEADER.")
    print(f"Optimal Parameters: Repulsion Radius={best_parameters[0]:.2f}, Repulsion Weight={best_parameters[1]:.2f}, Param A={best_parameters[2]:.2f}, Param B={best_parameters[3]:.2f}, Param C={best_parameters[4]:.2f}")