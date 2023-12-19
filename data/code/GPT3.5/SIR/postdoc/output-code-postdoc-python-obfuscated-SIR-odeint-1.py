import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dS/dt, dI/dt and dR/dt

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

# Initial conditions

S0 = 0.99
I0 = 0.01
R0 = 0.0
y0 = [S0, I0, R0]

# Parameters (beta and gamma)

beta = 0.35
gamma = 0.1

# Time vector

t = np.linspace(0, 100, 1000)

# Solve the SIR model equations

sol = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]

# Plotting

plt.figure(figsize=(10,6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend(loc='best')
plt.grid()
plt.show()
