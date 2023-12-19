from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def SIR_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_SIR_model(beta, gamma, initial_conditions, t_span):
    S0, I0, R0 = initial_conditions
    y0 = [S0, I0, R0]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(SIR_model, t_span, y0, args=(beta, gamma), t_eval=t_eval, method='RK45')
    return sol.t, sol.y


def plot_SIR_model(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


beta = 0.5
gamma = 0.1
initial_conditions = [0.99, 0.01, 0]
t_span = [0, 50]

t, y = simulate_SIR_model(beta, gamma, initial_conditions, t_span)
S, I, R = y

plot_SIR_model(t, S, I, R)
