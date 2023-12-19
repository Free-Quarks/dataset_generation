import numpy as np
import matplotlib.pyplot as plt


def euler_sir_model(beta, gamma, N, I0, R0, t): 
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = t[1] - t[0]
    for i in range(1, len(t)): 
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S.append(S[i-1] + dt * dS)
        I.append(I[i-1] + dt * dI)
        R.append(R[i-1] + dt * dR)
    return S, I, R


# Example usage:

# Parameters
beta = 0.2  # infection rate
gamma = 0.1  # recovery rate
N = 1000  # total population
I0 = 1  # initial number of infected individuals
R0 = 0  # initial number of recovered individuals
t = np.linspace(0, 100, 100)  # time points

# Run the model
S, I, R = euler_sir_model(beta, gamma, N, I0, R0, t)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using Euler Method')
plt.show()
