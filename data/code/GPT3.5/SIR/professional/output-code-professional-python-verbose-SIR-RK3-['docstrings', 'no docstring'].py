import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, t_max, S0, I0, R0):
    """
    Simulates the SIR model using the Runge-Kutta 3rd order method.
    
    Parameters:
    - beta (float): rate of transmission
    - gamma (float): recovery rate
    - t_max (float): maximum time
    - S0 (float): initial susceptible population
    - I0 (float): initial infected population
    - R0 (float): initial recovered population
    
    Returns:
    - t (numpy.ndarray): time array
    - S (numpy.ndarray): array of susceptible population over time
    - I (numpy.ndarray): array of infected population over time
    - R (numpy.ndarray): array of recovered population over time
    """
    
    def f(t, y):
        """
        Function representing the SIR model
        """
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])
    
    def rk3_step(f, t, y, dt):
        """
        Runge-Kutta 3rd order method step
        """
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt, y - dt * k1 + 2 * dt * k2)
        return y + dt/6 * (k1 + 4 * k2 + k3)
    
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps + 1)
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    y = np.array([S0, I0, R0])
    
    for i in range(num_steps):
        S[i], I[i], R[i] = y
        y = rk3_step(f, t[i], y, dt)
    
    S[-1], I[-1], R[-1] = y
    
    return t, S, I, R


# Example usage
beta = 0.2
gamma = 0.1
t_max = 100
S0 = 1000
I0 = 1
R0 = 0

t, S, I, R = sir_model(beta, gamma, t_max, S0, I0, R0)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

