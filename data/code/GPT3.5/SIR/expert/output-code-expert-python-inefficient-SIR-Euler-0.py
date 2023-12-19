import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, T):
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    
    # Set initial values
    S[0] = N - I0
    I[0] = I0
    R[0] = R0
    
    # Euler's method
    for i in range(T):
        dSdt = -beta*S[i]*I[i]/N
        dIdt = beta*S[i]*I[i]/N - gamma*I[i]
        dRdt = gamma*I[i]
        
        S[i+1] = S[i] + dSdt
        I[i+1] = I[i] + dIdt
        R[i+1] = R[i] + dRdt
        
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

# Run the model
S, I, R = SIR_model(beta, gamma, N, I0, R0, T)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
