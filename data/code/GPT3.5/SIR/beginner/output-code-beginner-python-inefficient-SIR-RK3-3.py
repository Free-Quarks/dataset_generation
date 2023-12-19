import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, R0, T):
    def SIR_model(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.linspace(0, T, T+1)
    y0 = N - I0 - R0
    y = np.array([y0, I0, R0])
    
    sol = []
    for i in range(T):
        t_cur = np.array([t[i], t[i] + (t[i+1] - t[i])/2, t[i+1]])
        h = t_cur[1] - t_cur[0]
        k1 = np.array(SIR_model(y, t_cur[0]))
        k2 = np.array(SIR_model(y + h/3 * k1, t_cur[1]))
        k3 = np.array(SIR_model(y + 2*h/3 * k2, t_cur[2]))
        y = y + h/4 * (k1 + 3*k3)
        sol.append(y)
    sol = np.array(sol)
    
    plt.plot(t, sol[:, 0], label='S')
    plt.plot(t, sol[:, 1], label='I')
    plt.plot(t, sol[:, 2], label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model with RK3')
    plt.legend(loc='best')
    plt.show()
}

