import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, I0, R0, E0, T):
    S = N - I0 - R0 - E0
    I = I0
    R = R0
    E = E0

    S_list = [S]
    I_list = [I]
    R_list = [R]
    E_list = [E]

    dt = 1
    num_steps = int(T / dt)

    for i in range(num_steps):
        S_new = S - beta * S * I / N
        I_new = I + (beta * S * I / N) - gamma * I - sigma * I
        R_new = R + gamma * I
        E_new = E + sigma * I

        S = S_new
        I = I_new
        R = R_new
        E = E_new

        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        E_list.append(E)

    return S_list, E_list, I_list, R_list


N = 1000
beta = 0.2
gamma = 0.1
sigma = 0.05
I0 = 1
R0 = 0
E0 = 0
T = 100

S, E, I, R = seir_model(N, beta, gamma, sigma, I0, R0, E0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
