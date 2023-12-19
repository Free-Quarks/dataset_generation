import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S, I, R, beta, gamma, t_max, dt):
    N = S + I + R
    S_values = [S]
    I_values = [I]
    R_values = [R]
    t_values = [0]

    for t in range(t_max):
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I

        S += dS_dt * dt
        I += dI_dt * dt
        R += dR_dt * dt

        S_values.append(S)
        I_values.append(I)
        R_values.append(R)
        t_values.append(t * dt)

    return S_values, I_values, R_values, t_values


S = 1000
I = 1
R = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

S_values, I_values, R_values, t_values = SIR_model(S, I, R, beta, gamma, t_max, dt)

plt.plot(t_values, S_values, label='Susceptible')
plt.plot(t_values, I_values, label='Infected')
plt.plot(t_values, R_values, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
