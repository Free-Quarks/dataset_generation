
    import numpy as np
    import matplotlib.pyplot as plt
    import json

    def model(t, y, beta, gamma):
        S, I, R = y
        return np.array([-beta*S*I, beta*S*I - gamma*I, gamma*I])

    def RK3(y, h, t, beta, gamma):
        k1 = h * model(t, y, beta, gamma)
        k2 = h * model(t + h/2, y + k1/2, beta, gamma)
        k3 = h * model(t + h, y + k2, beta, gamma)
        return (k1 + 4*k2 + k3) / 6

    def simulate():
        N = 1000
        I0 = 1
        S0 = N - I0
        R0 = 0
        beta = 0.2
        gamma = 0.1
        T = 160
        dt = 0.1
        steps = int(T/dt) + 1
        t = np.linspace(0, T, steps)
        y = np.empty((3, steps))
        y[:, 0] = [S0, I0, R0]
        for step in range(steps-1):
            y[:, step+1] = y[:, step] + RK3(y[:, step], dt, t[step], beta, gamma)
        plt.figure(figsize=(6,4))
        plt.plot(t, y[0, :], label="S(t)")
        plt.plot(t, y[1, :], label="I(t)")
        plt.plot(t, y[2, :], label="R(t)")
        plt.legend()
        plt.show()

    simulate()
    
