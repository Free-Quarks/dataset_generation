from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def serid_rk3(S0, E0, I0, R0, beta, sigma, gamma, t_end, num_points):
    def deriv(t, y):
        S, E, I, R = y
        dSdt = -beta(t) * S * I
        dEdt = beta(t) * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    y0 = [S0, E0, I0, R0]
    t_span = [0, t_end]
    t_eval = np.linspace(0, t_end, num_points)

    solution = solve_ivp(deriv, t_span, y0, t_eval=t_eval)

    return solution


beta = lambda t: 0.2
sigma = 0.5
gamma = 0.1
t_end = 100
t_num_points = 1000
S0, E0, I0, R0 = 1000, 1, 0, 0

solution = serid_rk3(S0, E0, I0, R0, beta, sigma, gamma, t_end, t_num_points)

plt.figure()
plt.plot(solution.t, solution.y[0], label='Susceptible')
plt.plot(solution.t, solution.y[1], label='Exposed')
plt.plot(solution.t, solution.y[2], label='Infected')
plt.plot(solution.t, solution.y[3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model using RK3')
plt.legend()
plt.show()
