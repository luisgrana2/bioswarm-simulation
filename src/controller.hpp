#pragma once

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "mrs_multirotor_simulator/uav_system/multirotor_model.hpp"

using State = mrs_multirotor_simulator::MultirotorModel::State;
using VelocityHdg = mrs_multirotor_simulator::reference::VelocityHdg;

class Controller
{
 public:
    explicit Controller(const std::vector<double> &parameters, size_t id)
        : id_(id), leader_initialized_(false), current_dynamic_leader_(0)
    {
        mode_ = parameters.size() > 0 ? static_cast<int>(parameters.at(0)) : 0;
        max_speed_ = parameters.size() > 1 ? parameters.at(1) : 6.0;
        rep_radius_ = parameters.size() > 2 ? parameters.at(2) : 12.0;
        rep_weight_ = parameters.size() > 3 ? parameters.at(3) : 4.0;
        param_a_ = parameters.size() > 4 ? parameters.at(4) : 0.8; 
        param_b_ = parameters.size() > 5 ? parameters.at(5) : 1.2; 
        param_c_ = parameters.size() > 6 ? parameters.at(6) : 1.5; 
        
        target_x_ = 0.0;
        target_y_ = 0.0;
        target_z_ = 10.0;
    }

    void set_target(double x, double y, double z)
    {
        target_x_ = x;
        target_y_ = y;
        target_z_ = z;
    }

    VelocityHdg step(
        double dt, size_t my_id, const std::vector<State>& all_states
    )
    {
        State my_state = all_states[my_id];
        Eigen::Vector3d pos = my_state.x;
        Eigen::Vector3d target(target_x_, target_y_, target_z_);

        Eigen::Vector3d v_repulsion(0, 0, 0);

        for (size_t i = 0; i < all_states.size(); ++i) {
            if (i == my_id) continue;
            Eigen::Vector3d vec_away = pos - all_states[i].x;
            double dist = vec_away.norm();
            if (dist > 0 && dist < rep_radius_) {
                v_repulsion += (vec_away / dist) * (rep_radius_ - dist);
            }
        }
        if (v_repulsion.norm() > 0.0001) {
            v_repulsion.normalize();
        }

        Eigen::Vector3d desired_vel(0, 0, 0);
        size_t active_leader_id = 0;

        if (mode_ == 0) {
            Eigen::Vector3d center_of_mass(0, 0, 0);
            for (const auto& s : all_states) center_of_mass += s.x;
            center_of_mass /= all_states.size();

            Eigen::Vector3d v_cohesion = center_of_mass - pos;
            if (v_cohesion.norm() > 0.0001) v_cohesion.normalize();
            
            Eigen::Vector3d v_target = target - pos;
            if (v_target.norm() > 0.0001) v_target.normalize();

            Eigen::Vector3d blend = v_cohesion * param_a_ + v_target * param_b_ + v_repulsion * rep_weight_;
            if (blend.norm() > 0.0001) blend.normalize();
            
            desired_vel = blend * max_speed_;
        }
        else if (mode_ == 1) {
            active_leader_id = 0;
            if (my_id == active_leader_id) {
                Eigen::Vector3d center_of_mass(0, 0, 0);
                for (const auto& s : all_states) center_of_mass += s.x;
                center_of_mass /= all_states.size();

                Eigen::Vector3d v_tether = center_of_mass - pos;
                double leash_distance = 8.0;
                if (v_tether.norm() > leash_distance) {
                    v_tether.normalize();
                } else {
                    v_tether = Eigen::Vector3d(0, 0, 0);
                }

                Eigen::Vector3d v_target = target - pos;
                if (v_target.norm() > 0.0001) v_target.normalize();

                Eigen::Vector3d blend = v_target * param_c_ + v_tether * param_a_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * (max_speed_ - 0.5);
            } else {
                Eigen::Vector3d v_leader = all_states[active_leader_id].x - pos;
                if (v_leader.norm() > 0.0001) v_leader.normalize();
                
                Eigen::Vector3d blend = v_leader * param_b_ + v_repulsion * rep_weight_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * max_speed_;
            }
        }
        else if (mode_ == 2) {
            if (!leader_initialized_) {
                double min_dist = (all_states[0].x - target).norm();
                current_dynamic_leader_ = 0;
                for (size_t i = 1; i < all_states.size(); ++i) {
                    double d = (all_states[i].x - target).norm();
                    if (d < min_dist) {
                        min_dist = d;
                        current_dynamic_leader_ = i;
                    }
                }
                leader_initialized_ = true;
            } else {
                size_t candidate_leader = current_dynamic_leader_;
                double current_dist = (all_states[current_dynamic_leader_].x - target).norm();
                double min_dist = current_dist;

                for (size_t i = 0; i < all_states.size(); ++i) {
                    double d = (all_states[i].x - target).norm();
                    if (d < min_dist) {
                        min_dist = d;
                        candidate_leader = i;
                    }
                }

                if ((current_dist - min_dist) > 2.0) {
                    current_dynamic_leader_ = candidate_leader;
                }
            }

            active_leader_id = current_dynamic_leader_;

            if (my_id == active_leader_id) {
                Eigen::Vector3d center_of_mass(0, 0, 0);
                for (const auto& s : all_states) center_of_mass += s.x;
                center_of_mass /= all_states.size();

                Eigen::Vector3d v_tether = center_of_mass - pos;
                double leash_distance = 8.0;
                if (v_tether.norm() > leash_distance) {
                    v_tether.normalize();
                } else {
                    v_tether = Eigen::Vector3d(0, 0, 0);
                }

                Eigen::Vector3d v_target = target - pos;
                if (v_target.norm() > 0.0001) v_target.normalize();

                Eigen::Vector3d blend = v_target * param_c_ + v_tether * param_a_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * (max_speed_ - 0.5);
            } else {
                Eigen::Vector3d v_leader = all_states[active_leader_id].x - pos;
                if (v_leader.norm() > 0.0001) v_leader.normalize();
                
                Eigen::Vector3d blend = v_leader * param_b_ + v_repulsion * rep_weight_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * max_speed_;
            }
        }

        VelocityHdg cmd;
        cmd.velocity = desired_vel;
        cmd.heading = 0.0;
        return cmd;
    }

 private:
    size_t id_;
    int mode_;
    double rep_radius_;
    double rep_weight_;
    double param_a_;
    double param_b_;
    double param_c_;
    double target_x_;
    double target_y_;
    double target_z_;
    double max_speed_;
    
    bool leader_initialized_;
    size_t current_dynamic_leader_;
};