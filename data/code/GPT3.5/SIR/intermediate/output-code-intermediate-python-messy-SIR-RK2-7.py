def sir_rk2(N, beta, gamma, I0, T):
    import numpy as np
    import matplotlib.pyplot as plt

    h = T[1] - T[0]
    t = np.arange(T[0], T[1], h)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for i in range(1, len(t)):
        k1_s = -beta * S[i-1] * I[i-1] / N
        k1_i = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_s = -beta * (S[i-1] + 0.5 * h * k1_s) * (I[i-1] + 0.5 * h * k1_i) / N
        k2_i = beta * (S[i-1] + 0.5 * h * k1_s) * (I[i-1] + 0.5 * h * k1_i) / N - gamma * (I[i-1] + 0.5 * h * k1_i)

        S[i] = S[i-1] + h * k2_s
        I[i] = I[i-1] + h * k2_i
        R[i] = R[i-1] + gamma * (I[i-1] + 0.5 * h * k1_i)

    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model - RK2')
    plt.legend()
    plt.show()

sir_rk2(1000, 0.3, 0.1, 10, [0, 100])
