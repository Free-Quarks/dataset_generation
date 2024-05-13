import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Model parameters
gamma = 1/14  # Recovery rate
alpha = 0.2  # Asymptomatic rate
beta = 0.4   # Transmission rate
mu = 0.01    # Mortality rate
rho = 0.01   # Hospitalization rate (rate at which E compartments progress to H compartments)
sigma = 0.1  # ICU rate (rate at which H compartments progress to I compartments)


# Model equations
def model(x, t):
    S, I, D, A, R, T, H, E = x
    N = S + I + D + A + R + T + H + E

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (1 - alpha) * gamma * I - alpha * rho * I - mu * I
    dDdt = alpha * rho * I
    dAdt = (1 - alpha) * gamma * I
    dRdt = gamma * (1 - alpha) * I
    dTdt = gamma * alpha * I - sigma * T
    dHdt = sigma * T
    dEdt = mu * I

    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Initial conditions
S0 = 1000000  # Initial susceptible individuals
I0 = 1        # Initial infected individuals
D0 = 0        # Initial deceased individuals
A0 = 0        # Initial asymptomatic individuals
R0 = 0        # Initial recovered individuals
T0 = 0        # Initial ICU individuals
H0 = 0        # Initial hospitalized individuals
E0 = 0        # Initial exposed individuals


# Time vector
t = np.linspace(0, 365, 1000)


# Solve the model equations
y = odeint(model, [S0, I0, D0, A0, R0, T0, H0, E0], t)


# Plotting the results
plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Deceased')
plt.plot(t, y[:, 3], label='Asymptomatic')
plt.plot(t, y[:, 4], label='Recovered')
plt.plot(t, y[:, 5], label='ICU')
plt.plot(t, y[:, 6], label='Hospitalized')
plt.plot(t, y[:, 7], label='Exposed')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
