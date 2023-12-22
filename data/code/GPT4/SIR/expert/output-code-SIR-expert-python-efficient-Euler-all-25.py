import numpy as np
import matplotlib.pyplot as plt
import json

def simulate_sir(s0, i0, r0, beta, gamma, t):
    S, I, R = [s0], [i0], [r0]
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_s = S[-1] - (beta*S[-1]*I[-1])*dt
        next_i = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_r = R[-1] + (gamma*I[-1])*dt

        S.append(next_s)
        I.append(next_i)
        R.append(next_r)

    plt.figure(figsize=(10,5))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.legend()
    plt.show()

simulate_sir(999, 1, 0, 0.3, 0.1, np.linspace(0, 160, 1600))
