import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]

    while t[-1] < t_max:
        dS_dt = -beta * S[-1] * I[-1] / N
        dI_dt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR_dt = gamma * I[-1]

        S.append(S[-1] + dt * dS_dt)
        I.append(I[-1] + dt * dI_dt)
        R.append(R[-1] + dt * dR_dt)

        t.append(t[-1] + dt)

    return S, I, R

S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 50
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()

