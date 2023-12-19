import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SIR model

def sir_model(beta, gamma, N, I0, R0, days):
    # Total population
    S0 = N - I0 - R0
    
    # Initial conditions
    S = [S0]
    I = [I0]
    R = [R0]
    
    # Time vector
    t = np.linspace(0, days, days)
    
    # Step size
    h = t[1]-t[0]
    
    # RK4 integration loop
    for i in range(1, days):
        # Current values
        s = S[i-1]
        i = I[i-1]
        r = R[i-1]
        
        # Derivatives
        dsdt = -beta*s*i/N
        didt = beta*s*i/N - gamma*i
        drdt = gamma*i
        
        # RK4 steps
        k1s = h * dsdt
        k1i = h * didt
        k1r = h * drdt
        
        k2s = h * (-beta*(s+0.5*k1s)*(i+0.5*k1i)/N)
        k2i = h * (beta*(s+0.5*k1s)*(i+0.5*k1i)/N - gamma*(i+0.5*k1i))
        k2r = h * gamma*(i+0.5*k1i)
        
        k3s = h * (-beta*(s+0.5*k2s)*(i+0.5*k2i)/N)
        k3i = h * (beta*(s+0.5*k2s)*(i+0.5*k2i)/N - gamma*(i+0.5*k2i))
        k3r = h * gamma*(i+0.5*k2i)
        
        k4s = h * (-beta*(s+k3s)*(i+k3i)/N)
        k4i = h * (beta*(s+k3s)*(i+k3i)/N - gamma*(i+k3i))
        k4r = h * gamma*(i+k3i)
        
        # Next values
        S.append(s + (k1s + 2*k2s + 2*k3s + k4s)/6)
        I.append(i + (k1i + 2*k2i + 2*k3i + k4i)/6)
        R.append(r + (k1r + 2*k2r + 2*k3r + k4r)/6)
    
    return t, S, I, R


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
days = 100

# Run SIR model
t, S, I, R = sir_model(beta, gamma, N, I0, R0, days)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
