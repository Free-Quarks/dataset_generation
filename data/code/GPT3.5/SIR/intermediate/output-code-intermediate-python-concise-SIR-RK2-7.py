import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, beta, gamma, I0, T):
    def derivs(y, t):
        S,I,R = y
        dSdt = -beta*S*I/N
        dIdt = beta*S*I/N - gamma*I
        dRdt = gamma*I
        return [dSdt, dIdt, dRdt]

    t = np.linspace(0, T, T+1)
    y0 = [N-I0, I0, 0]
    sol = odeint(derivs, y0, t)

    S = sol[:, 0]
    I = sol[:, 1]
    R = sol[:, 2]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model Simulation')
    plt.show()

N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
T = 100

SIR_RK2(N, beta, gamma, I0, T)
