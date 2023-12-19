import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, t_max):
    # Define the ODEs
    def dS(S, I):
        return -beta * S * I
    
    def dI(S, I):
        return beta * S * I - gamma * I
    
    def dR(I):
        return gamma * I
    
    # Initialize arrays
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Define time step and array
    dt = 0.1
    t = np.arange(0, t_max, dt)
    
    # Solve the ODEs using the RK4 method
    for i in range(t_max - 1):
        dS1 = dt * dS(S[i], I[i])
        dI1 = dt * dI(S[i], I[i])
        dR1 = dt * dR(I[i])
        
        dS2 = dt * dS(S[i] + 0.5 * dS1, I[i] + 0.5 * dI1)
        dI2 = dt * dI(S[i] + 0.5 * dS1, I[i] + 0.5 * dI1)
        dR2 = dt * dR(I[i] + 0.5 * dI1)
        
        dS3 = dt * dS(S[i] + 0.5 * dS2, I[i] + 0.5 * dI2)
        dI3 = dt * dI(S[i] + 0.5 * dS2, I[i] + 0.5 * dI2)
        dR3 = dt * dR(I[i] + 0.5 * dI2)
        
        dS4 = dt * dS(S[i] + dS3, I[i] + dI3)
        dI4 = dt * dI(S[i] + dS3, I[i] + dI3)
        dR4 = dt * dR(I[i] + dI3)
        
        S[i + 1] = S[i] + (dS1 + 2 * dS2 + 2 * dS3 + dS4) / 6
        I[i + 1] = I[i] + (dI1 + 2 * dI2 + 2 * dI3 + dI4) / 6
        R[i + 1] = R[i] + (dR1 + 2 * dR2 + 2 * dR3 + dR4) / 6
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
t_max = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion of Population')
plt.title('SIR Model')
plt.legend()
plt.show()
