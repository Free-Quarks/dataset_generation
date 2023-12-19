import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, days):
    # Initial conditions
    S0 = N - I0 - R0
    I = np.zeros(days)
    R = np.zeros(days)
    S = np.zeros(days)
    
    I[0] = I0
    R[0] = R0
    S[0] = S0
    
    # Solve the differential equations using RK3 method
    for i in range(1, days):
        S[i] = S[i-1] - (beta * S[i-1] * I[i-1] / N)
        I[i] = I[i-1] + (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        R[i] = R[i-1] + (gamma * I[i-1])
    
    # Plot the results
    plt.plot(range(days), S, label='Susceptible')
    plt.plot(range(days), I, label='Infected')
    plt.plot(range(days), R, label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
days = 100

SIR_model(beta, gamma, N, I0, R0, days)
