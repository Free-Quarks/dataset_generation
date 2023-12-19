import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, days, days)
    dt = t[1] - t[0]
    for i in range(1, days):
        S.append(S[i-1] - beta * S[i-1] * I[i-1] * dt)
        I.append(I[i-1] + (beta * S[i-1] * I[i-1] - gamma * I[i-1]) * dt)
        R.append(R[i-1] + gamma * I[i-1] * dt)
    return S, I, R

# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
days = 100

# Run the model
S, I, R = SIR_model(beta, gamma, N, I0, R0, days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
