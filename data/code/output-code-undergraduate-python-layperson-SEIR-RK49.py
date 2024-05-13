import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1/7  # Recovery rate
sigma = 1/5  # Incubation rate
beta = 0.5    # Contact rate

# Initial conditions
S0 = 999  # Susceptible individuals
E0 = 1    # Exposed individuals
I0 = 0    # Infected individuals
R0 = 0    # Recovered individuals

# Time vector
t = np.linspace(0, 100, 1000)

# Function implementing the SEIR model
def seir_model(y, t):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Solve the SEIR model using RK4
from scipy.integrate import solve_ivp
def seir_solver():
    y0 = S0, E0, I0, R0
    solution = solve_ivp(seir_model, t_span=(0, 100), y0=y0, t_eval=t, method='RK45')
    S, E, I, R = solution.y
    return S, E, I, R

# Plot the results
S, E, I, R = seir_solver()
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model')
plt.show()
