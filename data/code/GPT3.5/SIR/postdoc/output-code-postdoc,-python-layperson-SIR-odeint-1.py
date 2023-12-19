import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function implementing the SIR model
def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Function to solve the differential equations
def solve_SIR_model(N, I0, R0, beta, gamma, t):    
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, t, t)
    solution = odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S, I, R = solution.T
    return S, I, R

# Parameters
N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t = 160

# Solve the model
S, I, R = solve_SIR_model(N, I0, R0, beta, gamma, t)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

