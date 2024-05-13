import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def simulate_seir_model(beta, sigma, gamma, N, E0, I0, R0, duration):
    t_span = (0, duration)
    y0 = N - E0 - I0 - R0, E0, I0, R0
    t_eval = np.linspace(0, duration, 1000)
    sol = solve_ivp(seir_model, t_span, y0, method='RK45', t_eval=t_eval, args=(beta, sigma, gamma))
    t = sol.t
    S, E, I, R = sol.y
    return t, S, E, I, R


beta = 0.5
sigma = 0.1
gamma = 0.2
N = 1000
E0 = 10
I0 = 1
R0 = 0
duration = 100


t, S, E, I, R = simulate_seir_model(beta, sigma, gamma, N, E0, I0, R0, duration)

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
