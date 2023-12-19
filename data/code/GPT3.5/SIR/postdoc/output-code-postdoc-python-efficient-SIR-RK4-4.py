import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(N, beta, gamma, days):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, days, days)
    y0 = N-1, 1, 0
    res = np.zeros((days, 3))
    res[0] = y0
    for i in range(1, days):
        dt = t[i] - t[i-1]
        k1 = deriv(res[i-1], t[i-1], N, beta, gamma)
        k2 = deriv(res[i-1] + 0.5 * dt * k1, t[i-1] + 0.5 * dt, N, beta, gamma)
        k3 = deriv(res[i-1] + 0.5 * dt * k2, t[i-1] + 0.5 * dt, N, beta, gamma)
        k4 = deriv(res[i-1] + dt * k3, t[i], N, beta, gamma)
        res[i] = res[i-1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    S, I, R = res[:, 0], res[:, 1], res[:, 2]
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of people')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()
}
