import numpy as np
import matplotlib.pyplot as plt


def sir_rk3(N, beta, gamma, t_max, dt, I_0):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0] = N - I_0
    I[0] = I_0
    R[0] = 0
    
    # Runge-Kutta 3rd order integration
    for i in range(1, len(t)):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dI_dt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR_dt = gamma * I[i-1]
        
        S1 = S[i-1] + dt * dS_dt
        I1 = I[i-1] + dt * dI_dt
        R1 = R[i-1] + dt * dR_dt
        
        dS1_dt = -beta * S1 * I1 / N
        dI1_dt = beta * S1 * I1 / N - gamma * I1
        dR1_dt = gamma * I1
        
        S2 = 3/4 * S[i-1] + 1/4 * S1 + 1/4 * dt * dS1_dt
        I2 = 3/4 * I[i-1] + 1/4 * I1 + 1/4 * dt * dI1_dt
        R2 = 3/4 * R[i-1] + 1/4 * R1 + 1/4 * dt * dR1_dt
        
        dS2_dt = -beta * S2 * I2 / N
        dI2_dt = beta * S2 * I2 / N - gamma * I2
        dR2_dt = gamma * I2
        
        S[i] = 1/3 * S[i-1] + 2/3 * S2 + 2/3 * dt * dS2_dt
        I[i] = 1/3 * I[i-1] + 2/3 * I2 + 2/3 * dt * dI2_dt
        R[i] = 1/3 * R[i-1] + 2/3 * R2 + 2/3 * dt * dR2_dt
    
    return t, S, I, R


# Example usage
N = 100000
beta = 0.35
gamma = 0.1
t_max = 100
dt = 0.1
I_0 = 100

t, S, I, R = sir_rk3(N, beta, gamma, t_max, dt, I_0)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
