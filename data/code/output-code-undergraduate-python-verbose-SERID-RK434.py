import numpy as np
import matplotlib.pyplot as plt


def serid_model(beta, gamma, N, I0, E0, R0, t_max):
    def deriv(y, t, beta, gamma, N):
        S, E, R, I, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - gamma * E
        dRdt = gamma * E
        dIdt = (1 - gamma) * E
        dDdt = gamma * E
        return dSdt, dEdt, dRdt, dIdt, dDdt

    t = np.linspace(0, t_max, t_max)
    y0 = N - I0 - E0 - R0, E0, R0, I0, 0
    ret = odeint(deriv, y0, t, args=(beta, gamma, N))
    S, E, R, I, D = ret.T
    
    return t, S, E, R, I, D


beta = 0.2
gamma = 0.1
N = 1000
I0, E0, R0 = 1, 0, 0
t_max = 100

t, S, E, R, I, D = serid_model(beta, gamma, N, I0, E0, R0, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, R, label='Recovered')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model Simulation')
plt.legend()
plt.show()
