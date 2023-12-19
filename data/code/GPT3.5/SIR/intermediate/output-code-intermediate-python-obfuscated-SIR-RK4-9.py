from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

# Define the SIR model


# Define the differential equations

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Define the initial conditions

S0 = 1000
I0 = 1
R0 = 0

# Define the parameters

beta = 0.2
gamma = 0.1

# Define the time grid

t = np.linspace(0, 100, 1000)

# Create an instance of the ODE solver

solver = ode(sir_model)
solver.set_initial_value([S0, I0, R0], t[0])
solver.set_f_params(beta, gamma)

# Solve the system of equations

solution = np.zeros((len(t), 3))
solution[0] = [S0, I0, R0]

for i in range(1, len(t)):
    solution[i] = solver.integrate(t[i])

# Plot the results

plt.plot(t, solution[:, 0], label='S')
plt.plot(t, solution[:, 1], label='I')
plt.plot(t, solution[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
