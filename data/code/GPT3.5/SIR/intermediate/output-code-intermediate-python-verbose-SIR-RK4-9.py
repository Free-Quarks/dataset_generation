import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, t_max, dt):
    def derivs(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    t = np.arange(0, t_max+dt, dt)
    y = np.zeros((len(t), 3))
    y[0] = y0
    
    for i in range(len(t)-1):
        k1 = dt * derivs(t[i], y[i])
        k2 = dt * derivs(t[i] + 0.5*dt, y[i] + 0.5*k1)
        k3 = dt * derivs(t[i] + 0.5*dt, y[i] + 0.5*k2)
        k4 = dt * derivs(t[i] + dt, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    S, I, R = y.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    
    plt.show()

SIR_RK4(0.3, 0.1, 1000, 1, 0, 100, 0.1)
