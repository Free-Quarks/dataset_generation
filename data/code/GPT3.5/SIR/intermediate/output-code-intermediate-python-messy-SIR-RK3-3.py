import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIR model

def sir_model(initial_conditions, beta, gamma, t_max, dt):
    N = initial_conditions[0]
    I0 = initial_conditions[1]
    R0 = initial_conditions[2]
    S0 = N - I0 - R0
    
    S = [S0]
    I = [I0]
    R = [R0]
    
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    
    for i in range(len(t)-1):
        dS_dt = -beta * S[i] * I[i] / N
        dI_dt = beta * S[i] * I[i] / N - gamma * I[i]
        dR_dt = gamma * I[i]
        
        S.append(S[i] + dt * dS_dt)
        I.append(I[i] + dt * dI_dt)
        R.append(R[i] + dt * dR_dt)
    
    return t, S, I, R


# Example usage

initial_conditions = [1000, 1, 0]  # N, I0, R0
beta = 0.2
gamma = 0.1

t_max = 100
dt = 0.1


t, S, I, R = sir_model(initial_conditions, beta, gamma, t_max, dt)

plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
