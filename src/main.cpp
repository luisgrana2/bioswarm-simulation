#include <format>
#include <iostream>
#include <vector>

#include "controller.hpp"
#include "mrs_multirotor_simulator/uav_system/multirotor_model.hpp"
#include "mrs_multirotor_simulator/uav_system/uav_system.hpp"
#include "params.hpp"

struct Uav
{
    mrs_multirotor_simulator::UavSystem system;
    Controller controller;
};

class Simulator
{
 public:
    explicit Simulator(
        SimParameters sim_params, const std::vector<double> &controller_params, 
        const std::vector<Eigen::Vector3d> &start_positions,
        const std::vector<Eigen::Vector3d> &waypoints
    )
        : parameters_(std::move(sim_params)), waypoints_(waypoints), current_wp_idx_(0)
    {
        mode_ = controller_params.size() > 0 ? static_cast<int>(controller_params[0]) : 0;
        
        for (size_t i = 0; i < parameters_.starting_states.size() && i < start_positions.size(); ++i) {
            parameters_.starting_states[i].position = start_positions[i];
        }

        size_t id = 0;
        for (auto &&start_state : parameters_.starting_states)
        {
            uavs_.push_back({
                .system = mrs_multirotor_simulator::UavSystem(
                    mrs_multirotor_simulator::MultirotorModel::ModelParams(),
                    start_state.position,
                    start_state.heading
                ),
                .controller = Controller(controller_params, id++),
            });
            uavs_.back().system.makeStep(parameters_.dt);
            uavs_.back().system.makeStep(parameters_.dt);
        }
    }

    void run()
    {
        output_header_();
        for (size_t step = 0; step < parameters_.steps; ++step)
        {
            if (waypoints_.empty()) break;

            Eigen::Vector3d target = waypoints_[current_wp_idx_];
            
            std::vector<State> all_states;
            for (auto& uav : uavs_) {
                all_states.push_back(uav.system.getState());
            }

            double dist = 0.0;
            if (mode_ == 0) {
                Eigen::Vector3d com(0, 0, 0);
                for (const auto& s : all_states) com += s.x;
                com /= all_states.size();
                dist = (com - target).norm();
            } else if (mode_ == 1) {
                dist = (all_states[0].x - target).norm();
            } else if (mode_ == 2) {
                double min_dist = (all_states[0].x - target).norm();
                for (const auto& s : all_states) {
                    double d = (s.x - target).norm();
                    if (d < min_dist) min_dist = d;
                }
                dist = min_dist;
            }

            double target_threshold = 4.0;
            if (current_wp_idx_ < waypoints_.size() - 1) {
                if (dist < target_threshold) {
                    current_wp_idx_++;
                }
            } else {
                if (dist < target_threshold + 4.0) {
                    break;
                }
            }

            for (auto& uav : uavs_) {
                uav.controller.set_target(target.x(), target.y(), target.z());
            }

            step_controllers_();
            step_simulation_();
            
            if (step % 10 == 0) {
                output_step_();
            }
        }
    }

 private:
    void step_controllers_()
    {
        std::vector<State> all_states;
        for (auto& uav : uavs_) {
            all_states.push_back(uav.system.getState());
        }

        for (size_t uav_idx = 0; uav_idx < uavs_.size(); ++uav_idx)
        {
            auto command = uavs_[uav_idx].controller.step(
                parameters_.dt, uav_idx, all_states
            );
            uavs_[uav_idx].system.setInput(command);
        }
    }

    void step_simulation_()
    {
        for (auto &&uav : uavs_)
        {
            uav.system.makeStep(parameters_.dt);
        }
    }

    void output_header_()
    {
        std::cout << parameters_.starting_states.size() << "\n";
    }

    void output_step_()
    {
        for (auto &&uav : uavs_)
        {
            auto state = uav.system.getState();
            std::cout << state.x.transpose() << " ";
        }
        std::cout << "\n";
    }

    SimParameters parameters_;
    std::vector<Uav> uavs_;
    std::vector<Eigen::Vector3d> waypoints_;
    size_t current_wp_idx_;
    int mode_;
};

void parse_arguments(int argc, char **argv, std::vector<double>& controller_params, std::vector<Eigen::Vector3d>& start_positions, std::vector<Eigen::Vector3d>& waypoints)
{
    for (int i = 1; i <= 7; ++i) {
        if (argc > i) controller_params.push_back(std::stod(argv[i]));
    }
    
    if (argc > 9) {
        int num_drones = std::stoi(argv[8]);
        int num_waypoints = std::stoi(argv[9]);
        int arg_idx = 10;
        
        for (int i = 0; i < num_drones; ++i) {
            if (arg_idx + 2 < argc) {
                double x = std::stod(argv[arg_idx++]);
                double y = std::stod(argv[arg_idx++]);
                double z = std::stod(argv[arg_idx++]);
                start_positions.push_back(Eigen::Vector3d(x, y, z));
            }
        }
        
        for (int i = 0; i < num_waypoints; ++i) {
            if (arg_idx + 2 < argc) {
                double x = std::stod(argv[arg_idx++]);
                double y = std::stod(argv[arg_idx++]);
                double z = std::stod(argv[arg_idx++]);
                waypoints.push_back(Eigen::Vector3d(x, y, z));
            }
        }
    }
}

int main(int argc, char **argv)
try
{
    int num_drones = (argc > 8) ? std::stoi(argv[8]) : 20;
    auto sim_parameters = get_parameters(num_drones);
    std::vector<double> controller_parameters;
    std::vector<Eigen::Vector3d> start_positions;
    std::vector<Eigen::Vector3d> waypoints;
    
    parse_arguments(argc, argv, controller_parameters, start_positions, waypoints);

    auto sim = Simulator(sim_parameters, controller_parameters, start_positions, waypoints);
    sim.run();
}
catch (std::exception &e)
{
    std::cerr << std::format("Uncaught error:\n  {}\n", e.what());
    return 1;
}