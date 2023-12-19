import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, N, t)
    """
    Function to simulate the SIR model using RK4 method.
    """
    def derivative(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = S0, I0, R0
    t_span = (t[0], t[-1])
    t_eval = t
    sol = solve_ivp(derivative, t_span, y0, t_eval=t_eval, args=(beta, gamma))

    S, I, R = sol.y

    return S, I, R


def plot_SIR(S, I, R, t):
    """
    Function to plot the SIR model simulation.
    """
    fig, ax = plt.subplots()
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('SIR Model Simulation')
    ax.legend()
    plt.show()


# Example usage
t = np.linspace(0, 100, 1000)
S, I, R = SIR_model(beta=0.2, gamma=0.1, S0=900, I0=100, R0=0, N=1000, t=t)
plot_SIR(S, I, R, t)
