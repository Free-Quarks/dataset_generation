import numpy as np
import matplotlib.pyplot as plt

def sir_model_euler(S0, I0, R0, beta, gamma, t):
    dt = t[1] - t[0]
    SIR = np.zeros((3, t.size))
    SIR[:,0] = S0, I0, R0
    for i in range(1, t.size):
        S = SIR[0, i-1] - (beta*SIR[0, i-1]*SIR[1, i-1])*dt
        I = SIR[1, i-1] + (beta*SIR[0, i-1]*SIR[1, i-1] - gamma*SIR[1, i-1])*dt
        R = SIR[2, i-1] + (gamma*SIR[1, i-1])*dt
        SIR[:, i] = S, I, R
    return SIR

# Define initial conditions and parameters
S0, I0, R0 = 5000, 10, 0
beta, gamma = 0.5/5000, 1/7
t = np.linspace(0, 160, 160)

# Run simulation
SIR = sir_model_euler(S0, I0, R0, beta, gamma, t)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(t, SIR[0], label="Susceptible")
plt.plot(t, SIR[1], label="Infected")
plt.plot(t, SIR[2], label="Recovered")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("SIR Model")
plt.grid(True)
plt.show()
