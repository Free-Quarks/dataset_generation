import numpy as np
import matplotlib.pyplot as plt


# Function implementing the SIR model

def sir_model(beta, gamma, N, I0, R0, t_max):
    '''
    Simulate the SIR model using the Runge-Kutta 4th order method.
    Parameters:
        - beta: Contact rate (infection rate)
        - gamma: Recovery rate
        - N: Total population
        - I0: Initial number of infected individuals
        - R0: Initial number of recovered individuals
        - t_max: Maximum time
    Returns:
        - t: Array of time points
        - S: Array of susceptible individuals
        - I: Array of infected individuals
        - R: Array of recovered individuals
    '''
    
    def derivs(SIR, t, beta, gamma):
        S, I, R = SIR
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.linspace(0, t_max, t_max+1)
    
    # Initial conditions
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    # Solve the differential equations
    S, I, R = np.array([]), np.array([]), np.array([])
    S = np.append(S, S0)
    I = np.append(I, I0)
    R = np.append(R, R0)
    
    for i in range(1, t_max+1):
        SIR = S, I, R
        t_step = t[i] - t[i-1]
        dSdt1, dIdt1, dRdt1 = derivs(SIR, t[i-1], beta, gamma)
        dSdt2, dIdt2, dRdt2 = derivs(SIR + t_step/2 * np.array([dSdt1, dIdt1, dRdt1]), t[i-1] + t_step/2, beta, gamma)
        dSdt3, dIdt3, dRdt3 = derivs(SIR + t_step/2 * np.array([dSdt2, dIdt2, dRdt2]), t[i-1] + t_step/2, beta, gamma)
        dSdt4, dIdt4, dRdt4 = derivs(SIR + t_step * np.array([dSdt3, dIdt3, dRdt3]), t[i-1] + t_step, beta, gamma)
        S = np.append(S, S[i-1] + t_step/6 * (dSdt1 + 2*dSdt2 + 2*dSdt3 + dSdt4))
        I = np.append(I, I[i-1] + t_step/6 * (dIdt1 + 2*dIdt2 + 2*dIdt3 + dIdt4))
        R = np.append(R, R[i-1] + t_step/6 * (dRdt1 + 2*dRdt2 + 2*dRdt3 + dRdt4))
    
    return t, S, I, R


# Example usage

# Parameters
beta = 0.2
# Contact rate
gamma = 0.1
# Recovery rate
N = 1000
# Total population
I0 = 50
# Initial number of infected individuals
R0 = 0
# Initial number of recovered individuals
t_max = 100
# Maximum time

# Simulate SIR model
t, S, I, R = sir_model(beta, gamma, N, I0, R0, t_max)

# Plot the results
plt.figure()
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
