import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, I0, D0, A0, S0, R0, T, beta, gamma, delta, epsilon, theta, rho, kappa):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    # R0 and T represent the period of infectivity and the number of time steps.
    # beta, gamma, delta, epsilon, theta, rho, kappa are the model parameters.
    
    # Initial conditions
    S = S0
    I = I0
    D = D0
    A = A0
    R = R0
    
    # Step size
    dt = T[1] - T[0]
    
    # Empty arrays to store results
    S_array = np.empty_like(T)
    I_array = np.empty_like(T)
    D_array = np.empty_like(T)
    A_array = np.empty_like(T)
    R_array = np.empty_like(T)
    
    # Euler method
    for i, t in enumerate(T):
        # Calculate the rates of change
        dS = -beta * S * (I + delta * D + epsilon * A) / N
        dI = beta * S * (I + delta * D + epsilon * A) / N - (gamma + theta) * I
        dD = theta * I - (rho + kappa) * D
        dA = rho * D - gamma * A
        dR = gamma * (I + A) + kappa * D
        
        # Update the variables using the Euler method
        S += dt * dS
        I += dt * dI
        D += dt * dD
        A += dt * dA
        R += dt * dR
        
        # Store the results
        S_array[i] = S
        I_array[i] = I
        D_array[i] = D
        A_array[i] = A
        R_array[i] = R
    
    return S_array, I_array, D_array, A_array, R_array

# Example usage
N = 1000000  # Total population
I0 = 1000  # Initial number of infected individuals
D0 = 10  # Initial number of deceased individuals
A0 = 100  # Initial number of asymptomatically infected individuals
S0 = N - I0 - D0 - A0  # Initial number of susceptible individuals
R0 = 0  # Initial number of recovered individuals
T = np.linspace(0, 100, 100)  # Time grid
beta = 0.2  # Infection rate
gamma = 0.1  # Recovery rate
theta = 0.01  # Death rate
delta = 0.01  # Severe cases rate
epsilon = 0.1  # Asymptomatic cases rate
rho = 0.05  # Asymptomatic recovery rate
kappa = 0.1  # Deceased recovery rate

# Run the model
S_array, I_array, D_array, A_array, R_array = sidarthe_model(N, I0, D0, A0, S0, R0, T, beta, gamma, delta, epsilon, theta, rho, kappa)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(T, S_array, label='Susceptible')
plt.plot(T, I_array, label='Infected')
plt.plot(T, D_array, label='Deceased')
plt.plot(T, A_array, label='Asymptomatic')
plt.plot(T, R_array, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
