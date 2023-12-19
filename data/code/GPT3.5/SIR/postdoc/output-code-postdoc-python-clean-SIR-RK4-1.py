import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    def SIR(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.arange(0, t_end, dt)
    y0 = N - I0, I0, R0
    result = odeint(SIR, y0, t, args=(beta, gamma))
    S, I, R = result.T

    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

