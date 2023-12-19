import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, t_max):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, t_max, int(t_max))

    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        S.append(S[i-1] + dSdt)
        I.append(I[i-1] + dIdt)
        R.append(R[i-1] + dRdt)

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
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()

