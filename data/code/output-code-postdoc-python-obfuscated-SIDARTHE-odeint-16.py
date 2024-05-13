import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe_model(y, t, beta, epsilon, alpha, gamma, delta, theta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + epsilon * A) / N
    dIdt = beta * S * (I + epsilon * A) / N - alpha * I - gamma * I
    dDdt = delta * alpha * I
    dAdt = (1 - delta) * alpha * I - theta * A
    dRdt = gamma * I + theta * A
    dTdt = delta * alpha * I
    dHdt = theta * A
    dEdt = epsilon * beta * S * (I + epsilon * A) / N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, beta, epsilon, alpha, gamma, delta, theta, t):
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    params = (beta, epsilon, alpha, gamma, delta, theta)
    sol = odeint(sidarthe_model, y0, t, args=params)
    S, I, D, A, R, T, H, E = sol.T
    return S, I, D, A, R, T, H, E


S0 = 100000
I0 = 100
D0 = 10
A0 = 1000
R0 = 0
T0 = 10
H0 = 0
E0 = 1000
beta = 0.2
epsilon = 0.2
alpha = 0.1
gamma = 0.1
delta = 0.2
theta = 0.1
t = np.linspace(0, 100, 1000)

S, I, D, A, R, T, H, E = simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, beta, epsilon, alpha, gamma, delta, theta, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Active')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Tested')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
