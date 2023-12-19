import numpy as np
import matplotlib.pyplot as plt

# function to implement the SIR model

def SIR_model(beta, gamma, S0, I0, R0, N, tmax, dt):
    t = np.arange(0, tmax, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dSdt = -beta * I[i-1] * S[i-1] / N
        dIdt = (beta * I[i-1] * S[i-1] / N) - (gamma * I[i-1])
        dRdt = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
    
    return S, I, R

# parameters
beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0

tmax = 100
dt = 0.1

# calling the model function
S, I, R = SIR_model(beta, gamma, S0, I0, R0, N, tmax, dt)

# plot the results
t = np.arange(0, tmax, dt)
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
