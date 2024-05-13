import numpy as np
from scipy.integrate import odeint
from typing import Tuple


def serid_model(init_conditions: np.ndarray, time: np.ndarray, beta: float, gamma: float, sigma: float) -> np.ndarray:
    S0, E0, I0, R0, D0 = init_conditions
    N = S0 + E0 + I0 + R0 + D0
    
    def deriv(y, t):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        dDdt = 0  # Assuming no deaths
        return dSdt, dEdt, dIdt, dRdt, dDdt
    
    y0 = S0, E0, I0, R0, D0
    return odeint(deriv, y0, time)


# Example usage
init_conditions = np.array([1000, 1, 0, 0, 0])
time = np.linspace(0, 100, 100)
beta = 0.2
gamma = 0.1
sigma = 0.01

result = serid_model(init_conditions, time, beta, gamma, sigma)
print(result)
