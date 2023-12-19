import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, T, N):
    # Initial conditions
    S = [S0]
    I = [I0]
    R = [R0]
    t = np.linspace(0, T, N+1)
    dt = T/N
    
    for i in range(N):
        # Euler's method for the first step
        S_half = S[i] - beta*S[i]*I[i]*dt/2
        I_half = I[i] + (beta*S[i]*I[i] - gamma*I[i])*dt/2
        R_half = R[i] + gamma*I[i]*dt/2
        
        # Euler's method for the second step
        S_full = S[i] - beta*S_half*I_half*dt
        I_full = I[i] + (beta*S_half*I_half - gamma*I_half)*dt
        R_full = R[i] + gamma*I_half*dt
        
        # Append the values to the lists
        S.append(S_full)
        I.append(I_full)
        R.append(R_full)
        
    return t, S, I, R

# Parameter values
beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
T = 100
N = 1000

# Run the SIR model and plot the results
t, S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, T, N)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()
plt.title('SIR Model - RK2')
plt.show()
