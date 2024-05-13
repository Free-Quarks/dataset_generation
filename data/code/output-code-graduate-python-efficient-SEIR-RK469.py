import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SEIR model

def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    # Total population
    S0 = N - I0 - E0 - R0
    
    # Initial conditions
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    
    # Time vector
    t = np.linspace(0, T, T+1)
    
    # Step size
    h = t[1] - t[0]
    
    # Runge-Kutta 4th order method
    for i in range(T):
        
        # Compute derivatives
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]
        
        # Update values using RK4 method
        k1_S = h * dSdt
        k1_E = h * dEdt
        k1_I = h * dIdt
        k1_R = h * dRdt
        
        k2_S = h * (-beta * (S[i] + k1_S/2) * (I[i] + k1_I/2) / N)
        k2_E = h * (beta * (S[i] + k1_S/2) * (I[i] + k1_I/2) / N - sigma * (E[i] + k1_E/2))
        k2_I = h * (sigma * (E[i] + k1_E/2) - gamma * (I[i] + k1_I/2))
        k2_R = h * (gamma * (I[i] + k1_I/2))
        
        k3_S = h * (-beta * (S[i] + k2_S/2) * (I[i] + k2_I/2) / N)
        k3_E = h * (beta * (S[i] + k2_S/2) * (I[i] + k2_I/2) / N - sigma * (E[i] + k2_E/2))
        k3_I = h * (sigma * (E[i] + k2_E/2) - gamma * (I[i] + k2_I/2))
        k3_R = h * (gamma * (I[i] + k2_I/2))
        
        k4_S = h * (-beta * (S[i] + k3_S) * (I[i] + k3_I) / N)
        k4_E = h * (beta * (S[i] + k3_S) * (I[i] + k3_I) / N - sigma * (E[i] + k3_E))
        k4_I = h * (sigma * (E[i] + k3_E) - gamma * (I[i] + k3_I))
        k4_R = h * (gamma * (I[i] + k3_I))
        
        # Update values
        S.append(S[i] + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6)
        E.append(E[i] + (k1_E + 2*k2_E + 2*k3_E + k4_E) / 6)
        I.append(I[i] + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6)
        R.append(R[i] + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6)
    
    return t, S, E, I, R


# Example usage

# Parameters
beta = 0.2
sigma = 1/5
gamma = 1/10
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

# Call the function
t, S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, T)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()

