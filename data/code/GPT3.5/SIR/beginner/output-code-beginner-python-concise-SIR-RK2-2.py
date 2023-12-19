import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, T):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 0.01
    for t in np.arange(0, T, dt):
        dS_dt = -beta * S[-1] * I[-1] / N
        dI_dt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR_dt = gamma * I[-1]
        S_new = S[-1] + dt * dS_dt
        I_new = I[-1] + dt * dI_dt
        R_new = R[-1] + dt * dR_dt
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
    return S, I, R

S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
T = 50

S, I, R = SIR_model(S0, I0, R0, beta, gamma, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
