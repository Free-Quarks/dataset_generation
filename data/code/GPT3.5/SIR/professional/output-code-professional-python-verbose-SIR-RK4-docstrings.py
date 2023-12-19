import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, I0, N, t_max):
    """Simulates the SIR model using the fourth-order Runge-Kutta method.
    
    Parameters:
    beta (float): the infection rate
    gamma (float): the recovery rate
    I0 (float): initial number of infected individuals
    N (float): total population size
    t_max (float): maximum time
    
    Returns:
    t (array): array of time values
    S (array): array of susceptible individuals
    I (array): array of infected individuals
    R (array): array of recovered individuals
    """
    
    def f(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    t = np.linspace(0, t_max, t_max + 1)
    y0 = [N - I0, I0, 0]  # initial conditions
    
    h = t[1] - t[0]  # step size
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0], I[0], R[0] = y0
    for i in range(len(t) - 1):
        k1 = h * f(t[i], [S[i], I[i], R[i]])
        k2 = h * f(t[i] + h/2, [S[i] + k1[0]/2, I[i] + k1[1]/2, R[i] + k1[2]/2])
        k3 = h * f(t[i] + h/2, [S[i] + k2[0]/2, I[i] + k2[1]/2, R[i] + k2[2]/2])
        k4 = h * f(t[i] + h, [S[i] + k3[0], I[i] + k3[1], R[i] + k3[2]])
        
        S[i+1] = S[i] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        I[i+1] = I[i] + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
        R[i+1] = R[i] + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
        
    return t, S, I, R


beta = 0.2
gamma = 0.1
I0 = 1
N = 1000
t_max = 100

t, S, I, R = SIR_model(beta, gamma, I0, N, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
