import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIR model

def SIR_model(S, I, R, beta, gamma, N):
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Main function to simulate and plot the SIR model

def simulate_SIR_model(N, I0, beta, gamma, days):
    # Initial conditions
    S = N - I0
    R = 0
    
    # Empty lists to store the results
    S_values = [S]
    I_values = [I0]
    R_values = [R]
    
    # Euler's method to simulate the model
    for day in range(days):
        dSdt, dIdt, dRdt = SIR_model(S, I0, R, beta, gamma, N)
        S += dSdt
        I0 += dIdt
        R += dRdt
        
        S_values.append(S)
        I_values.append(I0)
        R_values.append(R)
        
    # Plotting the results
    plt.plot(S_values, label='Susceptible')
    plt.plot(I_values, label='Infected')
    plt.plot(R_values, label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()


# Example usage
N = 1000
I0 = 1
beta = 0.2
gamma = 0.1
days = 100

simulate_SIR_model(N, I0, beta, gamma, days)

