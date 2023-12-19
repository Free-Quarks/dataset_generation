import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, t_max, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    for i in range(1, int(t_max/dt)+1):
        S_new = S[-1] - (beta * S[-1] * I[-1] / N) * dt
        I_new = I[-1] + ((beta * S[-1] * I[-1]) / N - gamma * I[-1]) * dt
        R_new = R[-1] + gamma * I[-1] * dt
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        t.append(i * dt)
    return S, I, R, t

beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

S, I, R, t = SIR_model(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
