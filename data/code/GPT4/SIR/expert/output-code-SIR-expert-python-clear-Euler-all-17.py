import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, T, dt):
    N = S0 + I0 + R0
    S, I, R = [S0], [I0], [R0]
    t = np.arange(0, T, dt)

    for _ in t[1:]:
        next_S = S[-1] - dt * beta * S[-1] * I[-1] / N
        next_I = I[-1] + dt * (beta * S[-1] * I[-1] / N - gamma * I[-1])
        next_R = R[-1] + dt * gamma * I[-1]

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    plt.figure(figsize=(10,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.legend()
    plt.show()

    return S, I, R

# Example use
S, I, R = sir_model_euler(999, 1, 0, 0.2, 0.1, 200, 0.1)
