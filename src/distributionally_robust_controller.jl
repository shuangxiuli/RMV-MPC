#///////////////////////////////////////
#// File Name: distributionally_robust_convace_controller.jl
#// Author: shuangxiu Li (kr37@illinois.edu)
#// Date Created: 2025/04/13
#// Description: Distributionally Robust Controller
#///////////////////////////////////////

using DataStructures
using LinearAlgebra
using Printf
using Random
using RobotOS
import Convex: Variable, norm, quadform, minimize, dot, solve!
using ECOS
using Statistics
using DataFrames
using LinearAlgebra
using PyCall
using JuMP
using Interpolations
using JLD2


struct DRCControlParameter <: Parameter
    eamax::Float64  # Maximum abolute value of acceleration
    tcalc::Float64  # Allocated control computation time
    goal_pos::Vector{Float64} # [x, y] goal position
    dtr::Float64 # Replanning time interval
    dtc::Float64 # Euler integration time interval

    horizon::Int64 # Planning horizon    
    discount::Float64 # Discount factor

    human_size::Float64 # Human size

    epsilon::Float64 # Risk-sensitiveness parameter

    safety_distance::Float64

    max_ccp_iters::Int
    tol_ccp::Float64
    tol_goal::Float64

    function DRCControlParameter(eamax::Float64, tcalc::Float64, 
                                goal_pos::Vector{Float64}, dtr::Float64, dtc::Float64,
                                horizon::Int64,
                                discount::Float64,
                                human_size::Float64,                                
                                epsilon::Float64,
                                safety_distance::Float64,
                                max_ccp_iters::Int,
                                tol_ccp::Float64,
                                tol_goal::Float64)
        @assert eamax > 0.0 "eamax must be positive."
        @assert tcalc > 0.0 "tcalc must be positive."
        @assert dtr > 0.0 "dtr must be positive."
        @assert length(goal_pos) == 2 "goal_pos must be a 2D vector."
        return new(eamax, tcalc, goal_pos, dtr, dtc, horizon, discount, human_size,
                    epsilon,safety_distance,max_ccp_iters,tol_ccp,tol_goal)
    end
end

const CCPProfileRecord = NamedTuple{(:tcalc_actual, :ccp_iters,:status), Tuple{Float64, Int,String}}
mutable struct DRCController
    sim_param::SimulationParameter
    cnt_param::DRCControlParameter
    predictor::Predictor
    cost_param::DRCCostParameter

    schedule_position::Union{Nothing, DataFrame}
    schedule_control::Union{Nothing, DataFrame}

    prediction_dict::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result::Union{Nothing, SimulationResult}
    tcalc_actual::Union{Nothing, Float64}

    prediction_task::Union{Nothing, Task}
    control_update_task::Union{Nothing, Task}

    prediction_dict_tmp::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result_tmp::Union{Nothing, SimulationResult}
    tcalc_actual_tmp::Union{Nothing, Float64}
    u_value_tmp::Union{Nothing, Vector{Float64}}
    u_value::Union{Nothing, Vector{Float64}}

    previous_cnt_plan::Union{Nothing, Array{Float64, 2}}

    Goal_reached::Bool 
    ccp_profile::Vector{CCPProfileRecord}
end

function DRCController(sim_param::SimulationParameter,
                        cnt_param::DRCControlParameter,
                        predictor::Predictor,
                        cost_param::DRCCostParameter)

    ccp_profile = CCPProfileRecord[]
    sizehint!(ccp_profile, 10_000)

    return DRCController(sim_param, cnt_param, predictor, cost_param, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,false,
                         ccp_profile)
end

# main control functions below
function control!(controller::DRCController,
                    current_time::Time,
                    log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing);
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (current_time, msg))
        end
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end

    if !isnothing(controller.control_update_task) &&
        istaskdone(controller.control_update_task)
        if !isnothing(log)
            msg = "New Distributionally Robust control is available to the controller"
            push!(log, (current_time, msg))
        end
        controller.tcalc_actual = copy(controller.tcalc_actual_tmp);
        controller.u_value = copy(controller.u_value_tmp);
        controller.control_update_task = nothing;
    end
    u = copy(controller.u_value);
    if !isnothing(log)
        msg = "control: $(u) is applied to the system."
        push!(log, (current_time, msg))
    end
    return u
end

function adjust_old_prediction!(controller::DRCController,
                                previous_ado_pos_dict::Dict{String, Vector{Float64}},
                                latest_ado_pos_dict::Dict{String, Vector{Float64}})
    @assert keys(controller.prediction_dict) == keys(previous_ado_pos_dict)

    ado_agents_removed = setdiff(keys(previous_ado_pos_dict),
                                    keys(latest_ado_pos_dict));
    ado_agents_added = setdiff(keys(latest_ado_pos_dict),
                                keys(previous_ado_pos_dict));
    keys_list = collect(keys(controller.prediction_dict))
    for key in keys_list
        if in(key, ado_agents_removed)
            # If this ado_agent is removed in latest_ado_pos_dict, then remove
            # it from prediction_dict.
            pop!(controller.prediction_dict, key)
        else
            if typeof(controller.predictor) != OraclePredictor # skip this step for OraclePredictor as predictions are perfect.
                # If this ado_agent is existent in both previous and latest
                # ado_pos_dict, then reuse and adjust the previous prediction.
                (diff_x, diff_y) = latest_ado_pos_dict[key] -
                                    previous_ado_pos_dict[key];
                controller.prediction_dict[key][:, :, 1] .+= diff_x;
                controller.prediction_dict[key][:, :, 2] .+= diff_y;
            end
        end
    end
    for key in ado_agents_added
        # Treat this newly added ado_agent as a static obstacle
        # (until the new prediction becomes available to the controller)
        num_controls = 1;
        controller.prediction_dict[key] =
            zeros(controller.sim_param.num_samples*num_controls,
                    controller.sim_param.prediction_steps, 2);
        controller.prediction_dict[key][:, :, 1] .+= latest_ado_pos_dict[key][1];
        controller.prediction_dict[key][:, :, 2] .+= latest_ado_pos_dict[key][2];
    end
    @assert keys(controller.prediction_dict) == keys(latest_ado_pos_dict)
end

function schedule_prediction!(controller::DRCController,
                                ado_pos_dict::Dict,
                                run_id::Int64,
                                previous_ado_pos_dict::Union{Nothing, Dict{String, Vector{Float64}}}=nothing,
                                e_init::Union{Nothing, RobotState}=nothing);
    if !isnothing(previous_ado_pos_dict)
        adjust_old_prediction!(controller, previous_ado_pos_dict,
                                convert_nodes_to_str(reduce_to_positions(ado_pos_dict)));
    end
    
    if typeof(controller.predictor) == TrajectronPredictor
        controller.prediction_task = @task begin
            if controller.predictor.param.use_robot_future
                @assert !isnothing(e_init) "e_init must be given."
                robot_present_and_future=
                    get_robot_present_and_future(e_init,
                                                controller.u_schedule,
                                                controller.sim_param,
                                                controller.cnt_param);
            else
                robot_present_and_future = nothing;
            end
            controller.prediction_dict_tmp = 
                sample_future_ado_positions!(controller.predictor,
                                                ado_pos_dict,
                                                robot_present_and_future);
            # save_path = "./hotel_005/prediction_dict_tmp.jld2"
            # jldsave(save_path; prediction_dict_tmp=controller.prediction_dict_tmp)
            # @load "./hotel_005/prediction_dict_tmp.jld2" prediction_dict_tmp
            # controller.prediction_dict_tmp = prediction_dict_tmp
            # @load "./experiment_results/experiment_$(run_id)/prediction_dict_tmp.jld2" prediction_dict_tmp
            # controller.prediction_dict_tmp = prediction_dict_tmp
            # println("Loaded data: ", controller.prediction_dict_tmp) 
            if !controller.predictor.param.use_robot_future
                num_controls = 1
                # each value has to be (num_samples*num_controls, prediction_steps, 2) array
                for key in keys(controller.prediction_dict_tmp)
                    controller.prediction_dict_tmp[key] = 
                        repeat(controller.prediction_dict_tmp[key],
                                outer=(num_controls, 1, 1));
                end
            end
        end
   
    elseif typeof(controller.predictor) == GaussianPredictor || typeof(controller.predictor) == StopGaussianPredictor
        controller.prediction_task = @task begin
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor,
                                             convert_nodes_to_str(ado_pos_dict));
            num_controls = 1;
            # each value has to be (num_samples*num_controls, prediction_steps, 2) array
            for key in keys(controller.prediction_dict_tmp)
                controller.prediction_dict_tmp[key] =
                    repeat(controller.prediction_dict_tmp[key],
                           outer=(num_controls, 1, 1));
            end
        end
    else
        @error "Type of controller.predictor: $(typeof(controller.predictor)) is not supported."
    end
    schedule(controller.prediction_task)
end

function schedule_prediction_idx!(controller::DRCController,
                                ado_pos_dict::Dict,                                
                                current_sec::Float64,
                                run_id::Int64,
                                previous_ado_pos_dict::Union{Nothing, Dict{String, Vector{Float64}}}=nothing,
                                e_init::Union{Nothing, RobotState}=nothing);
    if !isnothing(previous_ado_pos_dict)
        adjust_old_prediction!(controller, previous_ado_pos_dict,
                                convert_nodes_to_str(reduce_to_positions(ado_pos_dict)));
    end

    if typeof(controller.predictor) == TrajectronPredictor
        controller.prediction_task = @task begin
            if controller.predictor.param.use_robot_future
                @assert !isnothing(e_init) "e_init must be given."
                robot_present_and_future=
                    get_robot_present_and_future(e_init,
                                                controller.u_schedule,
                                                controller.sim_param,
                                                controller.cnt_param);
            else
                robot_present_and_future = nothing;
            end
            controller.prediction_dict_tmp = 
                sample_future_ado_positions!(controller.predictor,
                                                ado_pos_dict,
                                                robot_present_and_future);
            # save_path = "./hotel_005/prediction_dict_tmp_$current_sec.jld2"
            # jldsave(save_path; prediction_dict_tmp=controller.prediction_dict_tmp)
            # @load "./hotel_005/prediction_dict_tmp_$current_sec.jld2" prediction_dict_tmp
            # controller.prediction_dict_tmp = prediction_dict_tmp
            # @load "./experiment_results/experiment_$(run_id)/prediction_dict_tmp_$current_sec.jld2" prediction_dict_tmp
            # controller.prediction_dict_tmp = prediction_dict_tmp
            # println("Loaded data: ", controller.prediction_dict_tmp) 
            if !controller.predictor.param.use_robot_future
                num_controls = 1
                # each value has to be (num_samples*num_controls, prediction_steps, 2) array
                for key in keys(controller.prediction_dict_tmp)
                    controller.prediction_dict_tmp[key] = 
                        repeat(controller.prediction_dict_tmp[key],
                                outer=(num_controls, 1, 1));
                end
            end
        end
   
    elseif typeof(controller.predictor) == GaussianPredictor || typeof(controller.predictor) == StopGaussianPredictor
        controller.prediction_task = @task begin
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor,
                                             convert_nodes_to_str(ado_pos_dict));
            num_controls = 1;
            # each value has to be (num_samples*num_controls, prediction_steps, 2) array
            for key in keys(controller.prediction_dict_tmp)
                controller.prediction_dict_tmp[key] =
                    repeat(controller.prediction_dict_tmp[key],
                           outer=(num_controls, 1, 1));
            end
        end
    else
        @error "Type of controller.predictor: $(typeof(controller.predictor)) is not supported."
    end
    schedule(controller.prediction_task)
end

function schedule_control_update!(controller::DRCController,
                                    w_init::WorldState,
                                    target_trajectory::Trajectory2D,
                                    ego_pos_goal_vec::Vector{Float64},
                                    dtc::Float64,
                                    target_speed::Float64, 
                                    safety_distance::Float64,
                                    max_MPC_iters::Int64;
                                    log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing);
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (w_init.t, msg))
        end
        # @load "./hotel_005/prediction_dict_history.jld2" prediction_dict_history
        # controller.prediction_dict = prediction_dict_history
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end
    if !isnothing(log)
        msg = "New Distributionally Robust control is scheduled."
        push!(log, (w_init.t, msg))
    end

    controller.control_update_task = @task begin
        controller.tcalc_actual_tmp, controller.u_value_tmp,controller.Goal_reached = 
            drc_control_update!(controller, controller.cnt_param,
                        ego_pos_goal_vec,dtc,target_speed,
                        controller.prediction_dict,w_init, 
                        safety_distance,max_MPC_iters);

    end
    schedule(controller.control_update_task)
end

# helper functions below
function drc_control_update!(controller::DRCController,
                                    cnt_param::DRCControlParameter,
                                    ego_pos_goal_vec::Vector{Float64},
                                    dtc::Float64,
                                    target_speed::Float64, 
                                    prediction_dict::Dict{String, Array{Float64, 3}},
                                    w_init::WorldState,
                                    safety_distance::Float64,
                                    max_MPC_iters::Int64
                                    )
                                    
    u = [0.0, 0.0]
    Goal_reached = false
    ccp_iters = 0
    status = "MMMM"

    tcalc_actual = 
        @elapsed u ,Goal_reached, ccp_iters,status = run_MPC!(controller,ego_pos_goal_vec,target_speed,
                                dtc,prediction_dict,safety_distance,
                                max_MPC_iters,cnt_param, w_init.e_state);

    push!(controller.ccp_profile,
      (tcalc_actual = tcalc_actual,
       ccp_iters    = ccp_iters,
       status       = String(status)))

    if tcalc_actual >= cnt_param.tcalc
        # tcalc actual has exceeded allowable computation time
        time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        @warn "$(time) [sec]: DRC computation took $(round(tcalc_actual, digits=3)) [sec], which exceeds the maximum computation time allowed."
    end

    # u = [0.0, 0.0]
    # println(u)

    return tcalc_actual, u , Goal_reached
end


function simulate_straight_motion(e_init::RobotState,
                            ego_pos_goal_vec::Vector{Float64}, 
                            cnt_param::DRCControlParameter,
                            target_speed::Float64, dtc::Float64)
    # Calculate the vector from the starting point to the target and its magnitude.
    start = e_init.x[1:2]
    dir = ego_pos_goal_vec - start
    norm_dir = norm(dir)
    # If the two points coincide, return a zero vector; otherwise calculate the unit direction vector.
    direction = norm_dir > 0 ? dir / norm_dir : [0.0, 0.0]
    
    # Compute the velocity vector based on the target speed.
    velocity = direction * target_speed
    
    # Calculate the distance from the start to the target and the total time required for movement.
    distance = norm( ego_pos_goal_vec - start)
    # total_time = distance / target_speed
    # n_steps = Int(ceil(total_time / dtc))
    
    # Initialize position tracking
    x_positions = [start[1]]
    y_positions = [start[2]]
    
    # Simulate linear motion
    current_pos = copy(start)
    for _ in 1:cnt_param.horizon
        current_pos += velocity * dtc
        # Check if the target position is reached or overshot to prevent overshooting
        if norm(current_pos - start) >= distance
            current_pos = ego_pos_goal_vec
            push!(x_positions, current_pos[1])
            push!(y_positions, current_pos[2])
            break
        end
        push!(x_positions, current_pos[1])
        push!(y_positions, current_pos[2])
    end
    
    # Construct a DataFrame to record position and velocity.
    schedule_position = DataFrame(x = x_positions, y = y_positions)
    schedule_control = DataFrame(vx = fill(velocity[1], length(x_positions)),
                            vy = fill(velocity[2], length(x_positions)))
    
    return schedule_position, schedule_control
end

function run_MPC!(controller::DRCController,
                        ego_pos_goal_vec::Vector{Float64}, 
                        target_speed::Float64, 
                        dtc::Float64,
                        prediction_dict::Dict{String, Array{Float64, 3}},
                        safety_distance::Float64,
                        max_MPC_iters::Int64,
                        cnt_param::DRCControlParameter,
                        e_init::RobotState
                        )

    u_candidates = nothing     
    
    schedule_position, schedule_control = simulate_straight_motion(e_init,ego_pos_goal_vec,cnt_param,target_speed,dtc)
    min_dist = Inf
    min_key = nothing
    Goal_reached = false
    ccp_iters = 0
    status="MM"

    for key in keys(prediction_dict)
        # Each trajectory is a (num_samples, num_predicted_timesteps, 2) array
        # traj = repeat(prediction_dict[key][1, :, :], inner=(4, 1))
        # traj = prediction_dict[key][1, :, :]
        # get mean
        agent_traj = dropdims(mean(prediction_dict[key], dims=1), dims=1)
        agent_traj_repeat = repeat(agent_traj, inner=(2, 1)) #inner=(2, 1)

        # dist = [ norm([schedule_position.x[i+1], schedule_position.y[i+1]] .- agent_traj_repeat[i,:])
        #     for i in 1:(nrow(schedule_position)-1) ]

        dist = [ norm([schedule_position.x[1], schedule_position.y[1]] .- agent_traj_repeat[1,:]) ]
        
        current_min = minimum(dist)
        if current_min < min_dist
            min_dist = current_min
            min_key = key                               
        end
    end
  
    u_candidates,cvar,ccp_iters,status = ccp_solver(controller, cnt_param, e_init, 
            ego_pos_goal_vec, min_key,prediction_dict) 


    if norm(e_init.x[1:2] - ego_pos_goal_vec) < cnt_param.tol_goal
        # println("Goal reached.")
        Goal_reached = true
        return u_candidates,Goal_reached,ccp_iters
    end     
    
    return u_candidates,Goal_reached,ccp_iters,status
end    

py"""
import cvxpy as cp
import numpy as np
import types
import sys
import json
import numpy as np

my_solver = types.ModuleType("my_solver")
def ccp_solver(controller,
                cnt_param,
                e_init,
                ego_pos_goal_vec,
                min_key,
                prediction_dict
                # u_can
                ):
    # print("ccp_solver begins")                 
    H = cnt_param.horizon
    dtr = cnt_param.dtr
    
    eps = cnt_param.epsilon
    tol_ccp = cnt_param.tol_ccp
    max_ccp_iters = cnt_param.max_ccp_iters

    #     data = json.load(f)
    #     sorted_keys = sorted(int(k) for k in data.keys())
    #     Omega_dict = {k: np.array(data[str(k)], dtype=np.float64) for k in sorted_keys}    

    # Omega_dict = {
    # 0: np.array([[0.05305249689066898, -0.033585436770738865, 0.020502063938711432],
    #              [-0.033585436770738865, 2.5165608803346435, -0.07789732044415604],
    #              [0.020502063938711432, -0.07789732044415604, 1.0]], dtype=np.float64),
    # 1: np.array([[0.13343459443676345, -0.05400297137104444, 0.030834357598716864],
    #              [-0.05400297137104444, 2.591052562568116, -0.06382935768787813],
    #              [0.030834357598716864, -0.06382935768787813, 1.0]], dtype=np.float64),
    # } #hotel big

    # Omega_dict = {
    # 0: np.array([[0.05305249689066898, -0.033585436770738865, 0.020502063938711432],
    #              [-0.033585436770738865, 2.5165608803346435, -0.07789732044415604],
    #              [0.020502063938711432, -0.07789732044415604, 1.0]], dtype=np.float64),
    # 1: np.array([[0.13343459443676345, -0.05400297137104444, 0.030834357598716864],
    #              [-0.05400297137104444, 2.591052562568116, -0.06382935768787813],
    #              [0.030834357598716864, -0.06382935768787813, 1.0]], dtype=np.float64),
    # } #hotel small

    # Omega_dict = {
    # 0: np.array([[0.06138864184620789, 0.007803333543303356, -0.05187618507253536],
    #              [0.007803333543303356, 0.04167727204986316, 0.013393944515501287],
    #              [-0.05187618507253536, 0.013393944515501287, 1.0]], dtype=np.float64),

    # 1: np.array([[0.33900374138424507, 0.042791857636988465, -0.11068705870104643],
    #              [0.042791857636988465, 0.13145250528365968, 0.02495215407073602],
    #              [-0.11068705870104643, 0.02495215407073602, 1.0]], dtype=np.float64),
    # } #eth

    Omega_dict = {
    0: np.array([[0.04568054, 0.00630403, -0.00698583],
                [0.006304031, 0.02551584, 0.00411031],
                [-0.00698583, 0.00411031, 1.0]], dtype=np.float64),

    1: np.array([[0.26007000, 0.03852235, -0.02149290],
                [0.03852235, 0.17087832, 0.020083393],
                [-0.02149290, 0.020083393, 1.0]], dtype=np.float64),
    } #eth


    # for k, v in Omega_dict.items():
    #     print(k, v)


    ado_traj = prediction_dict[min_key][0,:,:]    #The first prediction trajectory is arranged in time  
    pred_data = np.repeat(ado_traj, repeats=2, axis=0)  #repeats=2

    # Initial g_val calculation: initial guess using u_r = [0.0, 0.5] (modify if needed)
    start = np.asarray(list(e_init.x))[:2]
    
    # initial_g = start + np.array([0.0, 0.0]) * dtr - ado_traj[0]
    # initial_g = start + np.array([0.0, 0.5]) * dtr - ado_traj[0]

    # initial_g = start + u_can* dtr - ado_traj[0] 
    # initial_g = u_can

    initial_g = start - ado_traj[0] 
    g_val = [initial_g.copy() for _ in range(H)]

    prev_cost = np.inf
    ccp_iters_used = 0

    e_pos = cp.Variable((H+1, 2))
    u     = cp.Variable((H,   2))
    beta  = cp.Variable(H)
    M     = [cp.Variable((3, 3), PSD=True) for _ in range(H)]

    # CCP iterations
    for it in range(max_ccp_iters):   
        #  print("ccp iters:",it)

        constraints = []
        cost_expr = 0

        #  Initial state constraint: e_pos[0] equals x_current
        constraints += [e_pos[0, :] == start]
        constraints += [u >= -2, u <= 2]
        # constraints += [cp.sum_squares(u) <= 1]

        #  Construct constraints and cost for each time step k = 1,...,H
        for k in range(H):   
            adjusted_k = int(k // 2)        #repeat=2
            Omega = Omega_dict[adjusted_k]    
            constraints.append(e_pos[k+1, 0] == e_pos[k, 0] + dtr * u[k, 0])
            constraints.append(e_pos[k+1, 1] == e_pos[k, 1] + dtr * u[k, 1])

            # Define the expression: g_expr = (x[k] - ado_position_dict[k]) + (u[k] - u_i) * Delta_t            
            g_x = e_pos[k, 0] - pred_data[k, 0] + dtr * u[k, 0]
            g_y = e_pos[k, 1] - pred_data[k, 1] + dtr * u[k, 1]

            # g_x_val, g_y_val = g_val[k]
            # gx_prev, gy_prev = g_val[k]
            gx_prev = g_val[k][0]
            gy_prev = g_val[k][1]    
            # gx0, gy0 = g_val[k]
            # norm0 = np.hypot(gx0, gy0) + 1e-9
            # gx_prev = r * gx0 / norm0
            # gy_prev = r * gy0 / norm0 
            # gx_prev = 0.4 * gx0 / norm0
            # gy_prev = 0.4 * gy0 / norm0      

            lin_gx2 = gx_prev**2 + 2 * gx_prev * (g_x - gx_prev)
            lin_gy2 = gy_prev**2 + 2 * gy_prev * (g_y - gy_prev)       
 
            constraints.append(beta[k] + (1/eps) * cp.trace(Omega @ M[k]) <=0)            
  
            # Assemble A_k as an array
            row1 = cp.hstack([-1,     0,  g_x])
            row2 = cp.hstack([0,     -1,  g_y])
            row3 = cp.hstack([ g_x,  g_y,
                            0.16- lin_gx2 - lin_gy2 - beta[k] ])
            A_k = cp.vstack([row1, row2, row3])
            
            # SDP constraint: M[k] - A_k must be PSD (positive semidefinite)
            constraints.append(M[k] - A_k >> 0)

            # Accumulate cost: aim to drive the predicted state close to the goal (squared error)
            cost_expr += cp.sum_squares(e_pos[k+1, :] - ego_pos_goal_vec) 
            # cost_expr += (beta[k] + (1/eps) * cp.trace(Omega @ M[k]))            
            # cost_expr += (u[k, 0]**2+u[k, 1]**2)
            # cost_expr += cp.sum_squares(e_pos[k+1, :] - ego_pos_goal_vec)

        # Solution
        prob = cp.Problem(cp.Minimize(cost_expr), constraints)       
        # prob.solve(solver=cp.SCS, warm_start=True, eps_abs=1e-4, eps_rel=1e-4,verbose=False) 
        # prob.solve(solver=cp.MOSEK, warm_start=True, mosek_params={
        #        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-4},verbose=True)
        try:
            prob.solve(solver=cp.MOSEK, warm_start=True, mosek_params={
               "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-1},verbose=False)
            
        except cp.error.SolverError:
            return [0.0, 0.0], 0.0,ccp_iters_used,"NNNN"
        # except (cp.SolverError, Exception) as e:
        #     print(f"[Exception caught] {e}. Returning default value.")
        #     return [0.0, 0.0], 0.0
        # try:
        #     prob.solve(solver=cp.MOSEK, warm_start=True, mosek_params={"MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-4})
        # except cp.SolverError:
        #     prob.solve(solver=cp.SCS, warm_start=True, eps_abs=1e-4, eps_rel=1e-4,verbose=False) 


        ccp_iters_used = it + 1
        u_val   = u.value
        e_val   = e_pos.value
        cvar = (beta[0] + (1/eps) * cp.trace(Omega_dict[0] @ M[0])).value         
        #  print(prob.status,".....",prob.solver_stats.solve_time,".....",prob.solver_stats.num_iters)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # print(f"Iteration {it}: solver returned invalid values, aborting CCP.")
            return [0.0, 0.0],0.0,ccp_iters_used,prob.status
        else:
            # u_first = [float(u_val[0,0]), float(u_val[0,1])] 
            cvar_rounded = round(cvar, 3)
            if cvar_rounded >= 0:
                u_first = [0.0,0.0]
            else:
                u_first = [float(u_val[0,0]), float(u_val[0,1])]                 
                
        for k in range(H):
            # if u_val[k,:]!=[0.0,0.0]:
            gx_new = e_val[k,0] - pred_data[k,0] + dtr * u_val[k,0]
            gy_new = e_val[k,1] - pred_data[k,1] + dtr * u_val[k,1]
            g_val[k] = np.array([gx_new, gy_new])   

        curr_cost = prob.value

    #     # If the cost converges, exit the CCP iterations
        if it > 0 and abs(prev_cost - curr_cost) < tol_ccp:
            break
        prev_cost = curr_cost  

    return u_first, cvar,ccp_iters_used,prob.status

my_solver.ccp_solver = ccp_solver

sys.modules["my_solver"] = my_solver
"""
mod = pyimport("my_solver")
ccp_solver = mod["ccp_solver"]


# for Trajectron robot-future-conditoinal models
function get_robot_present_and_future(e_init::RobotState,
                                                u_schedule::OrderedDict{Time, Vector{Float64}},
                                                sim_param::SimulationParameter,
                                                cnt_param::DRCControlParameter)
    # compute u_arrays from u_schedule and nominal control candidates
    u_arrays = get_nominal_u_arrays(u_schedule, sim_param, cnt_param);
    # process u_arrays
    u_array_gpu = cu(process_u_arrays(u_arrays));
    # ego state simulation
    ex_array_gpu = simulate_forward(e_init, u_array_gpu, sim_param)
    slice_factor = Int64(round(sim_param.dto/sim_param.dtc, digits=5));
    ex_array_cpu = collect(ex_array_gpu[:, 1:slice_factor:end, :])
    @assert size(ex_array_cpu, 2) == sim_param.prediction_steps + 1

    # (num_controls, 1 + prediction_steps, 6) array where the last dimension
    # is [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]. accelerations are computed
    # by finite-differencing velocity snapshots.
    robot_present_and_future = Array{Float64, 3}(undef, size(ex_array_cpu, 1),
        size(ex_array_cpu, 2),
        6);
    robot_present_and_future[:, :, 1:4] = ex_array_cpu;
    acc = diff(ex_array_cpu[:, :, 3:4], dims=2)./sim_param.dto
    robot_present_and_future[:, 1:end-1, 5:6] = acc;
    # pad last acceleration
    robot_present_and_future[:, end, 5:6] = robot_present_and_future[:, end-1, 5:6]

    return robot_present_and_future
end

#///////////////////////////////////////
#// 使用正确的均值和方差；预测轨迹进行repeat，并且无解时不为0，改了目标函数。联合机会约束
#///////////////////////////////////////
