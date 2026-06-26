dtc = 0.2;                                                                         # Euler integration time interval [s]
dtr =  0.2;                                                                          # replanning time interval [s]
tcalc =  0.2;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 2.0;                                                                   # maximum control norm [m/s^2]

horizon = 3;
# horizon = 8;
discount = 0.99;

human_size = 0.4;
# human_size = 0.2;

epsilon = 0.15
# epsilon = [0.05,0.15,0.25]

# Omega = [155.6 15.2 15.1;
#         152 144.6 2.1;
#         15.1 2.1 1.0]

max_ccp_iters = 3
tol_ccp = 1e-1
tol_goal = 0.4

#safety_distance = 2.5
safety_distance = 1.0

max_MPC_iters = 250
