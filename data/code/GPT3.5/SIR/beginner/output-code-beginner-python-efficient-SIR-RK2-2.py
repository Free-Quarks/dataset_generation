import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SIR model

def SIR_model(beta, gamma, S0, I0, R0, t_end, N, h):
    
    # Initialize arrays
    t = np.arange(0, t_end, h)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta method
    for i in range(1, len(t)):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dI_dt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR_dt = gamma * I[i-1]
        
        S_half = S[i-1] + 0.5 * h * dS_dt
        I_half = I[i-1] + 0.5 * h * dI_dt
        R_half = R[i-1] + 0.5 * h * dR_dt
        
        dS_dt_half = -beta * S_half * I_half / N
        dI_dt_half = beta * S_half * I_half / N - gamma * I_half
        dR_dt_half = gamma * I_half
        
        S[i] = S[i-1] + h * dS_dt_half
        I[i] = I[i-1] + h * dI_dt_half
        R[i] = R[i-1] + h * dR_dt_half
        
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
N = 1000
h = 0.1
t_end = 100

# Run the SIR model
S, I, R = SIR_model(beta, gamma, S0, I0, R0, t_end, N, h)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
