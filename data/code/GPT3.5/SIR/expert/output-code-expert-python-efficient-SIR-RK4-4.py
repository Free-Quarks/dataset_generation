import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, t_max):
    # Define the differential equations
    def dSdt(S, I):
        return -beta * S * I
    
    def dIdt(S, I):
        return beta * S * I - gamma * I
    
    def dRdt(I):
        return gamma * I
    
    # Initialize arrays to store the results
    t = np.linspace(0, t_max, t_max + 1)
    S = np.zeros(t_max + 1)
    I = np.zeros(t_max + 1)
    R = np.zeros(t_max + 1)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Use Runge-Kutta 4th order method
    for i in range(t_max):
        h = t[i+1] - t[i]
        k1_S = dSdt(S[i], I[i])
        k1_I = dIdt(S[i], I[i])
        k1_R = dRdt(I[i])
        
        k2_S = dSdt(S[i] + 0.5 * h * k1_S, I[i] + 0.5 * h * k1_I)
        k2_I = dIdt(S[i] + 0.5 * h * k1_S, I[i] + 0.5 * h * k1_I)
        k2_R = dRdt(I[i] + 0.5 * h * k1_I)
        
        k3_S = dSdt(S[i] + 0.5 * h * k2_S, I[i] + 0.5 * h * k2_I)
        k3_I = dIdt(S[i] + 0.5 * h * k2_S, I[i] + 0.5 * h * k2_I)
        k3_R = dRdt(I[i] + 0.5 * h * k2_I)
        
        k4_S = dSdt(S[i] + h * k3_S, I[i] + h * k3_I)
        k4_I = dIdt(S[i] + h * k3_S, I[i] + h * k3_I)
        k4_R = dRdt(I[i] + h * k3_I)
        
        S[i+1] = S[i] + (1/6) * h * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        I[i+1] = I[i] + (1/6) * h * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        R[i+1] = R[i] + (1/6) * h * (k1_R + 2*k2_R + 2*k3_R + k4_R)
    
    # Plot the results
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}
