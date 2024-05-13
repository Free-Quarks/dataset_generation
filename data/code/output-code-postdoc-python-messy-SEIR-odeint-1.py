import numpy as np
from scipy.integrate import odeint


# Function implementing the SEIR model
def seir_model(y, t, beta, gamma, sigma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Function to solve the SEIR model

def solve_seir_model(y0, t, beta, gamma, sigma):
    N = np.sum(y0)
    args = (beta, gamma, sigma, N)
    solution = odeint(seir_model, y0, t, args)
    return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]
