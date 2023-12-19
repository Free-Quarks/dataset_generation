import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_SIR(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

# Initial conditions
S0 = 999
I0 = 1
R0 = 0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 100)

# Solve the ODEs
sol = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))

# Extracting S, I, R from the solution
S = sol[:, 0]
I = sol[:, 1]
R = sol[:, 2]

# Plot the results
plot_SIR(S, I, R, t)
