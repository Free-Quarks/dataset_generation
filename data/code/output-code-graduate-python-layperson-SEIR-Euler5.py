import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    E = [0]
    I = [I0]
    R = [R0]
    for t in range(days):
        S_to_E = (beta * S[t] * I[t]) / N
        E_to_I = sigma * E[t]
        I_to_R = gamma * I[t]
        S.append(S[t] - S_to_E)
        E.append(E[t] + S_to_E - E_to_I)
        I.append(I[t] + E_to_I - I_to_R)
        R.append(R[t] + I_to_R)
    return S, E, I, R

beta = 0.2
sigma = 0.5
gamma = 0.1
N = 100000
I0 = 100
R0 = 0
days = 200

S, E, I, R = seir_model(beta, gamma, sigma, N, I0, R0, days)

time = np.arange(days + 1)
plt.plot(time, S, label='Susceptible')
plt.plot(time, E, label='Exposed')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
