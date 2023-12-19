import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, t_end, dt):
    def derivs(y, t):
        S, I, R = y
        dS = -beta * S * I / N
        dI = (beta * S * I / N) - (gamma * I)
        dR = gamma * I
        return [dS, dI, dR]

    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    y0 = [N - I0 - R0, I0, R0]
    sol = np.zeros((len(t), 3))
    sol[0] = y0

    for i in range(len(t) - 1):
        k1 = derivs(sol[i], t[i])
        k2 = derivs(sol[i] + 0.5 * k1 * dt, t[i] + 0.5 * dt)
        k3 = derivs(sol[i] + 0.5 * k2 * dt, t[i] + 0.5 * dt)
        k4 = derivs(sol[i] + k3 * dt, t[i] + dt)
        sol[i+1] = sol[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt

    plt.figure()
    plt.plot(t, sol[:, 0], label='S')
    plt.plot(t, sol[:, 1], label='I')
    plt.plot(t, sol[:, 2], label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK4')
    plt.legend()
    plt.show()
}
