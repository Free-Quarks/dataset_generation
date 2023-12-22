import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, T, dt):
    """
    Simulates the SIR model using the Euler method.

    Parameters:
    S0, I0, R0: initial number of susceptible, infected and recovered individuals.
    beta: infection rate.
    gamma: recovery rate.
    T: total time.
    dt: time step.

    Returns:
    time: array of time points.
    S, I, R: arrays of the number of susceptible, infected and recovered individuals over time.
    """
    N = S0 + I0 + R0  # total population
    n_iter = int(T/dt)  # number of iterations
    time = np.arange(n_iter+1)*dt  # array of time points
    S = np.zeros(n_iter+1)
    I = np.zeros(n_iter+1)
    R = np.zeros(n_iter+1)
    S[0], I[0], R[0] = S0, I0, R0

    for i in range(n_iter):
        S[i+1] = S[i] - dt*beta*S[i]*I[i]/N
        I[i+1] = I[i] + dt*beta*S[i]*I[i]/N - dt*gamma*I[i]
        R[i+1] = R[i] + dt*gamma*I[i]

    return time, S, I, R

# Test the function
time, S, I, R = sir_model_euler(999, 1, 0, 0.3, 0.1, 100, 0.1)
plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.legend()
plt.show()

# Format the output as a JSON instance
output = {
    "code": """
import numpy as np
import matplotlib.pyplot as plt

def sir_model_euler(S0, I0, R0, beta, gamma, T, dt):
    N = S0 + I0 + R0
    n_iter = int(T/dt)
    time = np.arange(n_iter+1)*dt
    S = np.zeros(n_iter+1)
    I = np.zeros(n_iter+1)
    R = np.zeros(n_iter+1)
    S[0], I[0], R[0] = S0, I0, R0

    for i in range(n_iter):
        S[i+1] = S[i] - dt*beta*S[i]*I[i]/N
        I[i+1] = I[i] + dt*beta*S[i]*I[i]/N - dt*gamma*I[i]
        R[i+1] = R[i] + dt*gamma*I[i]

    return time, S, I, R
    "",
    "function_name": "sir_model_euler"
}

# Convert the output to JSON
output_json = json.dumps(output)
