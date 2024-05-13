import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SEIR model

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    for t in range(T):
        dSdt = -beta * S[t] * I[t] / N
        dEdt = beta * S[t] * I[t] / N - sigma * E[t]
        dIdt = sigma * E[t] - gamma * I[t]
        dRdt = gamma * I[t]
        S.append(S[t] + dSdt)
        E.append(E[t] + dEdt)
        I.append(I[t] + dIdt)
        R.append(R[t] + dRdt)
    return S, E, I, R

# Set the parameters
beta = 0.2
sigma = 0.1
gamma = 0.1
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

# Call the model function
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
