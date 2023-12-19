import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    # Initial conditions
    S0 = N - I0 - R0
    
    # Number of time steps
    num_steps = int(t_end/dt)
    
    # Arrays to store the results
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 2nd order method
    for i in range(1, num_steps):
        # Calculate the derivatives
        dSdt = -beta*S[i-1]*I[i-1]/N
        dIdt = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
        dRdt = gamma*I[i-1]
        
        # Calculate the intermediate values
        S_mid = S[i-1] + 0.5*dt*dSdt
        I_mid = I[i-1] + 0.5*dt*dIdt
        R_mid = R[i-1] + 0.5*dt*dRdt
        
        # Calculate the derivatives at the intermediate values
        dSdt_mid = -beta*S_mid*I_mid/N
        dIdt_mid = beta*S_mid*I_mid/N - gamma*I_mid
        dRdt_mid = gamma*I_mid
        
        # Update the values using the RK2 method
        S[i] = S[i-1] + dt*dSdt_mid
        I[i] = I[i-1] + dt*dIdt_mid
        R[i] = R[i-1] + dt*dRdt_mid
    
    # Return the arrays
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
t_end = 100
dt = 0.1

# Run the model
S, I, R = SIR_model(beta, gamma, N, I0, R0, t_end, dt)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
