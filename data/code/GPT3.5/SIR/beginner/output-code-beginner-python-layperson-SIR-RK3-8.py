import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, T):
    # Define the initial conditions
    S0 = N - I0 - R0
    
    # Define the time grid
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    
    # Define the arrays to store the solution
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    
    # Set the initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Implement the RK3 method
    for i in range(1, T+1):
        # Calculate the derivatives
        dSdt = -beta*S[i-1]*I[i-1]/N
        dIdt = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
        dRdt = gamma*I[i-1]
        
        # Calculate the intermediate values
        S_half = S[i-1] + 0.5*dt*dSdt
        I_half = I[i-1] + 0.5*dt*dIdt
        R_half = R[i-1] + 0.5*dt*dRdt
        
        # Calculate the derivatives using the intermediate values
        dSdt_half = -beta*S_half*I_half/N
        dIdt_half = beta*S_half*I_half/N - gamma*I_half
        dRdt_half = gamma*I_half
        
        # Calculate the next values
        S[i] = S[i-1] + dt*(2/3*dSdt + 1/3*dSdt_half)
        I[i] = I[i-1] + dt*(2/3*dIdt + 1/3*dIdt_half)
        R[i] = R[i-1] + dt*(2/3*dRdt + 1/3*dRdt_half)
        
    # Return the solution
    return S, I, R

# Define the parameters
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

# Run the simulation
S, I, R = SIR_RK3(beta, gamma, N, I0, R0, T)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
