import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIDARTHE model

def sidarthe_model(beta, sigma, gamma, alpha, delta, mu, population, initial_infected, total_days):
    # Calculate initial values
    E = initial_infected * 2
    I = initial_infected
    D = 0
    A = 0
    R = 0
    T = 0
    H = 0
    S = population - E - I - D - A - R - T - H
    
    # Create arrays to store values
    E_values = [E]
    I_values = [I]
    D_values = [D]
    A_values = [A]
    R_values = [R]
    T_values = [T]
    H_values = [H]
    S_values = [S]
    
    # Euler method to approximate the differential equations
    dt = 1
    for day in range(total_days):
        E_new = E + dt * (beta * S * I / population - sigma * E)
        I_new = I + dt * (sigma * E - (gamma + alpha + delta) * I)
        D_new = D + dt * (delta * I)
        A_new = A + dt * (alpha * I)
        R_new = R + dt * (gamma * I)
        T_new = T + dt * (mu * I)
        H_new = H + dt * (delta * I)
        S_new = population - E_new - I_new - D_new - A_new - R_new - T_new - H_new
        
        # Update values
        E = E_new
        I = I_new
        D = D_new
        A = A_new
        R = R_new
        T = T_new
        H = H_new
        S = S_new
        
        # Store values
        E_values.append(E)
        I_values.append(I)
        D_values.append(D)
        A_values.append(A)
        R_values.append(R)
        T_values.append(T)
        H_values.append(H)
        S_values.append(S)
        
    return S_values, E_values, I_values, D_values, A_values, R_values, T_values, H_values


# Example usage
population = 100000
initial_infected = 10
total_days = 100

beta = 0.8
sigma = 0.2
gamma = 0.1
alpha = 0.1
mu = 0.05
delta = 0.01

S, E, I, D, A, R, T, H = sidarthe_model(beta, sigma, gamma, alpha, delta, mu, population, initial_infected, total_days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(D, label='Dead')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Testing')
plt.plot(H, label='Hospitalized')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
