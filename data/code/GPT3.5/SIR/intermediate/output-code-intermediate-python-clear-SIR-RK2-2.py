import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, T):
    def dSdt(S, I):
        return - beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    def rk2_step(S, I, R, dt):
        S1 = S + dt * dSdt(S, I)
        I1 = I + dt * dIdt(S, I)
        R1 = R + dt * dRdt(I)
        S2 = S + dt * dSdt(S1, I1)
        I2 = I + dt * dIdt(S1, I1)
        R2 = R + dt * dRdt(I1)
        S = 0.5 * (S1 + S2)
        I = 0.5 * (I1 + I2)
        R = 0.5 * (R1 + R2)
        return S, I, R

    S = N - I0
    I = I0
    R = 0

    S_values = [S]
    I_values = [I]
    R_values = [R]

    dt = T / 100
    for _ in range(100):
        S, I, R = rk2_step(S, I, R, dt)
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)

    return S_values, I_values, R_values


beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
T = 10

S_values, I_values, R_values = SIR_RK2(beta, gamma, N, I0, T)

plt.plot(S_values, label='Susceptible')
plt.plot(I_values, label='Infected')
plt.plot(R_values, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SIR Model using RK2')
plt.show()

