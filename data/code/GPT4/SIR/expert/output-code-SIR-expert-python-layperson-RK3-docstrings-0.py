"""
Python program to simulate the SIR epidemiological model using the RK3 method.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_rk3(beta, gamma, S0, I0, R0, T, dt):
    """
    Simulate the SIR model using the RK3 method.

    Parameters:
    beta (float): Infection rate.
    gamma (float): Recovery rate.
    S0 (float): Initial susceptible population.
    I0 (float): Initial infected population.
    R0 (float): Initial recovered population.
    T (float): Total time to simulate for.
    dt (float): Time step for the simulation.

    Returns:
    dict: Time series data for the S, I, and R populations.
    """

    # Initial conditions
    S, I, R = S0, I0, R0
    S_list, I_list, R_list, time_list = [S0], [I0], [R0], [0]

    # RK3 method loop
    for _ in np.arange(dt, T+dt, dt):
        k1_S, k1_I, k1_R = dt * (-beta*S*I), dt * (beta*S*I - gamma*I), dt * (gamma*I)
        k2_S, k2_I, k2_R = dt * (-beta*(S+0.5*k1_S)*(I+0.5*k1_I)), dt * (beta*(S+0.5*k1_S)*(I+0.5*k1_I) - gamma*(I+0.5*k1_I)), dt * (gamma*(I+0.5*k1_I))
        S += (k1_S + 2*k2_S) / 3
        I += (k1_I + 2*k2_I) / 3
        R += (k1_R + 2*k2_R) / 3
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        time_list.append(_)

    # Plot SIR model
    plt.plot(time_list, S_list, label='S(t)')
    plt.plot(time_list, I_list, label='I(t)')
    plt.plot(time_list, R_list, label='R(t)')
    plt.legend()
    plt.show()

    return {"S": S_list, "I": I_list, "R": R_list, "time": time_list}
