import numpy as np
import matplotlib.pyplot as plt


def RK4(func, t0, y0, dt, N):
    t = np.zeros(N+1)
    y = np.zeros((N+1, len(y0)))
    t[0] = t0
    y[0] = y0
    
    for i in range(N):
        tn = t[i]
        yn = y[i]
        k1 = dt * func(tn, yn)
        k2 = dt * func(tn + dt/2, yn + k1/2)
        k3 = dt * func(tn + dt/2, yn + k2/2)
        k4 = dt * func(tn + dt, yn + k3)
        
        t[i+1] = tn + dt
        y[i+1] = yn + (k1 + 2*k2 + 2*k3 + k4) / 6
        
    return t, y


def SIDARTHE_model(t, y):
    S, I, D, A, R, T, H, E = y
    beta = 0.25
    gamma = 0.05
    alpha = 0.02
    rho = 0.01
    theta = 0.02
    eta = 0.01
    
    dS = -beta * S * (I + A + R + T + H + E)
    dI = beta * S * (I + A + R + T + H + E) - gamma * I - alpha * I
    dD = gamma * I - rho * D
    dA = alpha * I - theta * A - eta * A
    dR = rho * D
    dT = theta * A
    dH = eta * A
    dE = beta * S * (I + A + R + T + H + E)
    
    return [dS, dI, dD, dA, dR, dT, dH, dE]


t0 = 0
y0 = [1000000, 1, 0, 0, 0, 0, 0, 0]
dt = 0.01
N = 1000


t, y = RK4(SIDARTHE_model, t0, y0, dt, N)


plt.plot(t, y[:, 0], label='S')
plt.plot(t, y[:, 1], label='I')
plt.plot(t, y[:, 2], label='D')
plt.plot(t, y[:, 3], label='A')
plt.plot(t, y[:, 4], label='R')
plt.plot(t, y[:, 5], label='T')
plt.plot(t, y[:, 6], label='H')
plt.plot(t, y[:, 7], label='E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()
