import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, I0, R0, beta, gamma, duration):
    # Initial conditions
    S0 = N - I0 - R0
    Y0 = [S0, I0, R0]
    
    # Parameters
    h = 0.1
    n_steps = int(duration / h)
    
    # Arrays to store the results
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    t = np.zeros(n_steps)
    
    # Initialize the arrays
    S[0] = Y0[0]
    I[0] = Y0[1]
    R[0] = Y0[2]
    t[0] = 0
    
    # Run the model using the Runge-Kutta 3rd order method
    for i in range(1, n_steps):
        k1 = h * (-beta * S[i-1] * I[i-1] / N)
        k2 = h * (-beta * (S[i-1] + k1 / 2) * (I[i-1] + k1 / 2) / N)
        k3 = h * (-beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N)
        
        S[i] = S[i-1] + (k1 + 4 * k2 + k3) / 6
        k1 = h * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = h * (beta * (S[i-1] + k1 / 2) * (I[i-1] + k1 / 2) / N - gamma * (I[i-1] + k1 / 2))
        k3 = h * (beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N - gamma * (I[i-1] - k1 + 2 * k2))
        
        I[i] = I[i-1] + (k1 + 4 * k2 + k3) / 6
        R[i] = R[i-1] + h * gamma * (I[i-1] - k1 + 2 * k2)
        t[i] = t[i-1] + h
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()
}
