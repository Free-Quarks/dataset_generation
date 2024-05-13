import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SEIR model

def seir_model(beta, gamma, sigma, N, I0, E0, R0, days):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    dt = 0.1
    t = np.linspace(0, days, int(days/dt)+1)
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        S.append(S[i-1] + dt * dS)
        E.append(E[i-1] + dt * dE)
        I.append(I[i-1] + dt * dI)
        R.append(R[i-1] + dt * dR)
    return t, S, E, I, R

# Parameters
beta = 0.2
sigma = 0.1
gamma = 0.05
N = 1000
I0 = 1
E0 = 0
R0 = 0
days = 100

# Calling the SEIR model function
t, S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, days)

# Plotting the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()

