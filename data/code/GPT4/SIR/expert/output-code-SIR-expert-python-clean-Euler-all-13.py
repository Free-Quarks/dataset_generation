import numpy as np
import matplotlib.pyplot as plt
import json

# Define the main function that contains the SIR model dynamics
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Define the parameters and initial conditions
N = 1000
beta = 1.0
gamma = 0.5
S0, I0, R0 = N-1, 1, 0  # initial conditions
t = np.linspace(0, 49, 50)  # time grid

# Integrate the SIR equations over the time grid using Euler's method
y0 = S0, I0, R0  # initial conditions vector
dt = t[1] - t[0]  # time step
y = np.zeros((len(t), len(y0)))  # array for solution
y[0, :] = y0
for i in range(1, len(t)):
    y[i, :] = y[i-1, :] + dt * np.array(sir_model(y[i-1, :], i, beta, gamma))

# Plot the data
plt.plot(t, y[:, 0], 'b', label='S(t)')
plt.plot(t, y[:, 1], 'r', label='I(t)')
plt.plot(t, y[:, 2], 'g', label='R(t)')
plt.legend()
plt.show()

# Prepare the output as a JSON instance
code = open(__file__).read()
function_name = sir_model
output = {'code': code, 'function_name': function_name}
output_json = json.dumps(output)
