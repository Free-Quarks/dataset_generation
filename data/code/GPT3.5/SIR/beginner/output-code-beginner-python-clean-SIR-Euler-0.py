import numpy as np
import matplotlib.pyplot as plt


def SIR_model(N, beta, gamma, I0, t_end, step_size):
    # Initial conditions
    S0 = N - I0
    R0 = 0
    
    # Arrays to store the values
    S = np.zeros(t_end)
    I = np.zeros(t_end)
    R = np.zeros(t_end)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Euler's method
    for t in range(1, t_end):
        S[t] = S[t-1] - (beta * S[t-1] * I[t-1] / N) * step_size
        I[t] = I[t-1] + (beta * S[t-1] * I[t-1] / N - gamma * I[t-1]) * step_size
        R[t] = R[t-1] + gamma * I[t-1] * step_size
    
    # Plotting
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
N = 1000
beta = 0.2
gamma = 0.1
I0 = 1

t_end = 100
step_size = 0.1

SIR_model(N, beta, gamma, I0, t_end, step_size)
