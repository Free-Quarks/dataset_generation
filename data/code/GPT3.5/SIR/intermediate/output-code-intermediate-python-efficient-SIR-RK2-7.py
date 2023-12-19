import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, S0, I0, R0, N, t_end, dt):
    # Define the differential equations
    def dSdt(beta, S, I):
        return -beta * S * I / N
    
    def dIdt(beta, gamma, S, I):
        return beta * S * I / N - gamma * I
    
    def dRdt(gamma, I):
        return gamma * I
    
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Implement RK2 method
    for i in range(1, len(t)):
        k1_S = dt * dSdt(beta, S[i-1], I[i-1])
        k1_I = dt * dIdt(beta, gamma, S[i-1], I[i-1])
        k1_R = dt * dRdt(gamma, I[i-1])
        
        k2_S = dt * dSdt(beta, S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I)
        k2_I = dt * dIdt(beta, gamma, S[i-1] + 0.5 * k1_S, I[i-1] + 0.5 * k1_I)
        k2_R = dt * dRdt(gamma, I[i-1] + 0.5 * k1_I)
        
        S[i] = S[i-1] + k2_S
        I[i] = I[i-1] + k2_I
        R[i] = R[i-1] + k2_R
    
    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2 Method')
    plt.legend()
    plt.show()
}
