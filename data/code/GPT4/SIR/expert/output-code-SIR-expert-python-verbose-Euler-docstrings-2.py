import numpy as np
import matplotlib.pyplot as plt
import json
def euler_SIR(S0, I0, R0, beta, gamma, t):
    """
    Simulate an SIR epidemic using Euler's method.

    Parameters:
    S0 (int): Initial number of susceptible individuals
    I0 (int): Initial number of infected individuals
    R0 (int): Initial number of recovered individuals
    beta (float): Infection rate
    gamma (float): Recovery rate
    t (array): Timepoints to simulate over

    Returns:
    dict: A dictionary with keys 'S', 'I', 'R' mapping to arrays representing 
    the number of susceptible, infected, and recovered individuals at each timepoint.
    """

    S, I, R = [S0], [I0], [R0]
    dt = t[1] - t[0]

    for _ in t[1:]:
        next_S = S[-1] - beta*S[-1]*I[-1]*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + gamma*I[-1]*dt

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return {'S': S, 'I': I, 'R': R}
