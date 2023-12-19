import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_sir_model(t, S, I, R):
    plt.figure(figsize=(8, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


def simulate_sir_model(S0, I0, R0, beta, gamma, t_max, t_step):
    t = np.arange(0, t_max, t_step)
    y0 = [S0, I0, R0]
    sol = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]
    plot_sir_model(t, S, I, R)


# Example usage
def main():
    S0 = 999
    I0 = 1
    R0 = 0
    beta = 0.2
    gamma = 0.1
    t_max = 50
    t_step = 0.1
    simulate_sir_model(S0, I0, R0, beta, gamma, t_max, t_step)


if __name__ == '__main__':
    main()
