import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, t_max, N, dt):
    # Define the differential equations
    def dSdt(S, I, R):
        return -beta * S * I / N
    
    def dIdt(S, I, R):
        return beta * S * I / N - gamma * I
    
    def dRdt(S, I, R):
        return gamma * I
    
    # Initialize arrays to store values
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Implement RK3 method
    for i in range(1, len(t)):
        k1_S = dt * dSdt(S[i-1], I[i-1], R[i-1])
        k1_I = dt * dIdt(S[i-1], I[i-1], R[i-1])
        k1_R = dt * dRdt(S[i-1], I[i-1], R[i-1])
        
        k2_S = dt * dSdt(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)
        k2_I = dt * dIdt(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)
        k2_R = dt * dRdt(S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I, R[i-1] + 0.5 * k1_R)
        
        k3_S = dt * dSdt(S[i-1] - k1_S + 2 * k2_S, I[i-1] - k1_I + 2 * k2_I, R[i-1] - k1_R + 2 * k2_R)
        k3_I = dt * dIdt(S[i-1] - k1_S + 2 * k2_S, I[i-1] - k1_I + 2 * k2_I, R[i-1] - k1_R + 2 * k2_R)
        k3_R = dt * dRdt(S[i-1] - k1_S + 2 * k2_S, I[i-1] - k1_I + 2 * k2_I, R[i-1] - k1_R + 2 * k2_R)
        
        S[i] = S[i-1] + (1/6) * (k1_S + 4 * k2_S + k3_S)
        I[i] = I[i-1] + (1/6) * (k1_I + 4 * k2_I + k3_I)
        R[i] = R[i-1] + (1/6) * (k1_R + 4 * k2_R + k3_R)
    
    # Plot the results
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()
}
