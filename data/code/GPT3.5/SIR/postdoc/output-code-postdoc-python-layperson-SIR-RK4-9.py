import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, num_days):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, num_days, num_days+1)
    dt = t[1] - t[0]
    for i in range(num_days):
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]
        S.append(S[i] + dt * dSdt)
        I.append(I[i] + dt * dIdt)
        R.append(R[i] + dt * dRdt)
    return t, S, I, R


beta = 0.3
gamma = 0.1
S0 = 900
I0 = 100
R0 = 0
num_days = 100

t, S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, num_days)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
