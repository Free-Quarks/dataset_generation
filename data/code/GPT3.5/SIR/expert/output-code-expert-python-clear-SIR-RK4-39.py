import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(S, I, R, beta, gamma, dt):
    N = S + I + R
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    S_new = S + dt * dS
    I_new = I + dt * dI
    R_new = R + dt * dR
    return S_new, I_new, R_new


def simulate_SIR(S0, I0, R0, beta, gamma, dt, days):
    S = [S0]
    I = [I0]
    R = [R0]
    for _ in range(days):
        S_new, I_new, R_new = SIR_RK4(S[-1], I[-1], R[-1], beta, gamma, dt)
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
dt = 0.1
days = 100

S, I, R = simulate_SIR(S0, I0, R0, beta, gamma, dt, days)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
