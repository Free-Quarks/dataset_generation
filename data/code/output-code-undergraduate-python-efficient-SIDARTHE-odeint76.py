import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe_model(y, t, beta, gamma1, gamma2, alpha, theta, delta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - (gamma1 + gamma2) * I
    dDdt = delta * (gamma1 * I + gamma2 * I) - theta * D
    dAdt = (1 - delta) * (gamma1 * I + gamma2 * I) - alpha * A
    dRdt = gamma1 * I + gamma2 * I - theta * R
    dTdt = theta * (D + R)
    dHdt = alpha * A
    dEdt = beta * S * (I + alpha * A) / N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma1, gamma2, alpha, theta, delta, t):
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    params = (beta, gamma1, gamma2, alpha, theta, delta)
    sol = odeint(sidarthe_model, y0, t, args=params)
    S, I, D, A, R, T, H, E = sol.T
    return S, I, D, A, R, T, H, E


# Example usage
S0 = 1000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0
beta = 0.2
gamma1 = 0.1
gamma2 = 0.05
alpha = 0.5
theta = 0.1
delta = 0.1
t = np.linspace(0, 100, 1000)
S, I, D, A, R, T, H, E = simulate_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, beta, gamma1, gamma2, alpha, theta, delta, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deaths')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Transferred')
plt.plot(t, H, label='Hospitalized')
plt.plot(t, E, label='Exposed')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
