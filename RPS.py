# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:55:45 2024

@author: snagchowdh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the replicator equations for Rock-Paper-Scissors
def rps_ode(t, y):
    x_R, x_P, x_S = y
    pi_R = x_S - x_P
    pi_P = x_R - x_S
    pi_S = x_P - x_R
    
    avg_payoff = x_R * pi_R + x_P * pi_P + x_S * pi_S
    
    dx_R_dt = x_R * (pi_R - avg_payoff)
    dx_P_dt = x_P * (pi_P - avg_payoff)
    dx_S_dt = x_S * (pi_S - avg_payoff)
    
    return [dx_R_dt, dx_P_dt, dx_S_dt]

# Initial proportions of the strategies
x_R0 = 0.33 #np.random.uniform(0, 1/3)
x_P0 = 0.33 #np.random.uniform(0, 1/3)
x_S0 = 1.0 - x_R0 - x_P0
initial_conditions = [x_R0, x_P0, x_S0]

# Time span
t_span = (0, 5000)
t_eval = np.arange(0, 5000, 0.01)

# Solve the ODE using RKF45 method
solution = solve_ivp(rps_ode, t_span, initial_conditions, method='RK45', t_eval=t_eval)

# Extract the results
t = solution.t
x_R = solution.y[0]
x_P = solution.y[1]
x_S = solution.y[2]

# Select the final 50 units of time series
final_time_span = 50
t_final = t[-int(final_time_span / 0.01):]
x_R_final = x_R[-int(final_time_span / 0.01):]
x_P_final = x_P[-int(final_time_span / 0.01):]
x_S_final = x_S[-int(final_time_span / 0.01):]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_final, x_R_final, label='Rock', color='red')
plt.plot(t_final, x_P_final, label='Paper', color='green')
plt.plot(t_final, x_S_final, label='Scissors', color='blue')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('Evolution of Strategies in Rock-Paper-Scissors Game (Final 50 Time Units)')
plt.legend()
plt.grid(True)
plt.show()
