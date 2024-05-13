import matplotlib.pyplot as plt
import numpy as np


# Function to calculate derivatives


def derivs(state, t, p):
    # Unpacking variables
    S, I, D, A, R, T, H, E = state
    beta, alpha, gamma, delta, theta, mu, lambda_h, lambda_t, kappa = p

    # Calculating derivatives
    dSdt = -beta * S * (I + delta * A) / N
    dIdt = beta * S * (I + delta * A) / N - (alpha + gamma) * I
    dDdt = alpha * I - (lambda_h + mu + lambda_t) * D
    dAdt = theta * (alpha * I - (lambda_h + mu + lambda_t) * D) - (gamma + kappa) * A
    dRdt = gamma * I + gamma * A
    dTdt = lambda_t * D
    dHdt = lambda_h * D
    dEdt = delta * A - (theta + kappa) * E

    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


# Function to solve the model

def solve_model(p, state0, t0, t1):
    # Unpacking parameters
    beta, alpha, gamma, delta, theta, mu, lambda_h, lambda_t, kappa = p
    N = np.sum(state0)

    # Time steps
    dt = 1
    t = np.arange(t0, t1 + dt, dt)

    # Number of time steps
    n = len(t)

    # Array to store results
    results = np.zeros((n, len(state0)))
    results[0, :] = state0

    # Solve the model using RK3
    for i in range(1, n):
        t_i = t[i-1]
        state_i = results[i-1, :]

        # First step
        k1 = derivs(state_i, t_i, p)
        state_k1 = state_i + k1 * dt

        # Second step
        k2 = derivs(state_k1, t_i + dt, p)
        state_k2 = (3 * state_i + state_k1 + 3 * k1 * dt + k2 * dt) / 4

        # Third step
        k3 = derivs(state_k2, t_i + dt, p)
        state_k3 = (state_i + 2 * state_k1 + 3 * k2 * dt + 6 * k3 * dt) / 9

        # Update the state
        results[i, :] = state_k3

    return t, results
