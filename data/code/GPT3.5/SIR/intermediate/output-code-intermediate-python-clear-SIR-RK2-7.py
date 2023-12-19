import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, I0, R0, beta, gamma, t_max, dt):
    # Total population
    S0 = N - I0 - R0
    
    # Initial conditions
    S = [S0]
    I = [I0]
    R = [R0]
    
    # Time vector
    t = [0]
    
    # Run the simulation
    while t[-1] < t_max:
        # Calculate the derivatives
        dSdt = -beta * S[-1] * I[-1] / N
        dIdt = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dRdt = gamma * I[-1]
        
        # Update the variables using RK2 method
        S_new = S[-1] + dt * dSdt
        I_new = I[-1] + dt * dIdt
        R_new = R[-1] + dt * dRdt
        
        # Append the new values to the lists
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        
        # Increment the time
        t_new = t[-1] + dt
        t.append(t_new)
        
    # Return the results
    return S, I, R

# Parameters
N = 1000  # Total population
I0 = 1  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
beta = 0.2  # Infection rate
gamma = 0.1  # Recovery rate
T = 100  # Simulation time
dt = 0.1  # Time step

# Run the simulation
S, I, R = SIR_RK2(N, I0, R0, beta, gamma, T, dt)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
