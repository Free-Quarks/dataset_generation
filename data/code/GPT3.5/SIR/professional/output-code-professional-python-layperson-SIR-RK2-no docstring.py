import numpy as np
import matplotlib.pyplot as plt
def SIR_model(beta, gamma, S0, I0, R0, t_max):
    N = S0 + I0 + R0
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for t in range(1, t_max):
        S[t] = S[t-1] - beta * S[t-1] * I[t-1] / N
        I[t] = I[t-1] + beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        R[t] = R[t-1] + gamma * I[t-1]

    return S, I, R

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
