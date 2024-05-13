import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    # Total population
    S0 = N - I0 - E0 - R0
    # Initial conditions
    S, E, I, R = [S0], [E0], [I0], [R0]
    # Time vector
    t = np.linspace(0, T, T+1)
    dt = T / T
    # Run the model
    for i in range(T):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]
        S.append(S[i] + dt * dSdt)
        E.append(E[i] + dt * dEdt)
        I.append(I[i] + dt * dIdt)
        R.append(R[i] + dt * dRdt)
    # Return results
    return t, S, E, I, R

# Parameters
beta = 0.2
sigma = 1 / 5
gamma = 1 / 10
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

# Run SEIR model
t, S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, T)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
