import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, sigma, gamma, theta, delta, xi, mu, N, I0, R0, D0, T):
    def f(state, t):
        S, I, D, A, R, T, H, E = state
        N = S + I + D + A + R + T + H + E
        dS = -beta * S * (I + theta * A) / N
        dI = beta * S * (I + theta * A) / N - (sigma + gamma) * I
        dD = delta * sigma * I - xi * D
        dA = (1 - delta) * sigma * I - (mu + theta * gamma) * A
        dR = gamma * (I + theta * A) - xi * R
        dT = theta * gamma * (I + theta * A) - xi * T
        dH = mu * A - xi * H
        dE = xi * (D + R + T + H)
        return [dS, dI, dD, dA, dR, dT, dH, dE]

    state0 = [N-I0-R0-D0, I0, D0, 0, R0, T0, H0, E0]
    t = np.linspace(0, T, T+1)
    result = odeint(f, state0, t)
    result = np.array(result)
    S, I, D, A, R, T, H, E = result.T
    return S, I, D, A, R, T, H, E



beta = 0.3
sigma = 0.04
gamma = 0.07
theta = 0.03
delta = 0.03
xi = 0.04
mu = 0.01
N = 100000
I0 = 1
R0 = 0
D0 = 0
T = 200

S, I, D, A, R, T, H, E = sidarthe_model(beta, sigma, gamma, theta, delta, xi, mu, N, I0, R0, D0, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, I, label='Infected')
plt.plot(T, D, label='Deceased')
plt.plot(T, A, label='Asymptomatic')
plt.plot(T, R, label='Recovered')
plt.plot(T, T, label='Tested')
plt.plot(T, H, label='Hospitalized')
plt.plot(T, E, label='Estimated')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
