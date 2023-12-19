import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, T, int(T)+1)
    dt = t[1] - t[0]

    for i in range(len(t)-1):
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dRdt = gamma * I[-1]

        S.append(S[-1] + dt * dSdt)
        I.append(I[-1] + dt * dIdt)
        R.append(R[-1] + dt * dRdt)

    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
T = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.show()
