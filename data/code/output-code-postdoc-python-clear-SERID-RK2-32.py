import numpy as np
from scipy.integrate import odeint
from typing import Tuple


def serid_model(y: np.ndarray, t: float, beta: float, gamma: float, mu: float) -> np.ndarray:
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - mu * I
    dRdt = (1 - mu) * gamma * I
    dDdt = mu * I
    return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])


def simulate_serid(beta: float, gamma: float, mu: float, initial_conditions: Tuple[float, float, float, float, float], t: np.ndarray) -> np.ndarray:
    solution = odeint(serid_model, initial_conditions, t, args=(beta, gamma, mu))
    return solution


# Example usage

beta = 0.6
gamma = 0.2
mu = 0.05
initial_conditions = (1000, 10, 1, 0, 0)
t = np.linspace(0, 100, 100)
solution = simulate_serid(beta, gamma, mu, initial_conditions, t)

print(solution)
