import matplotlib.pyplot as plt
import numpy as np
"""Documentation
This script simulates the SIR dynamics using RK2 method.

"""

__a = 'RK2_SIR'

def b(S: float, I: float, R: float, beta: float, gamma: float) -> tuple:
    """Calculate dS, dI, dR"""
    return (-beta * S * I, beta * S * I - gamma * I, gamma * I)

def RK2_SIR(S: float, I: float, R: float, beta: float, gamma: float, dt: float):
    """Simulation of SIR dynamics using RK2 method"""
    dS1, dI1, dR1 = b(S, I, R, beta, gamma)
    dS2, dI2, dR2 = b(S + dt / 2 * dS1, I + dt / 2 * dI1, R + dt / 2 * dR1, beta, gamma)
    return S + dt * dS2, I + dt * dI2, R + dt * dR2

# Initial conditions
S, I, R = 0.9, 0.1, 0
beta, gamma = 0.35, 0.1
dt = 0.01

arr_S, arr_I, arr_R = [S], [I], [R]
for _ in range(1000):
    S, I, R = RK2_SIR(S, I, R, beta, gamma, dt)
    arr_S.append(S)
    arr_I.append(I)
    arr_R.append(R)

plt.figure(figsize=(10, 6))
plt.plot(arr_S, label='S')
plt.plot(arr_I, label='I')
plt.plot(arr_R, label='R')
plt.legend()
plt.show()
