import numpy as np
import matplotlib.pyplot as plt


# Function to simulate SIR model using Euler's method

def simulate_SIR(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]
    t = [0]
    
    for i in range(int(t_max/dt)):
        S.append(S[i] - beta*S[i]*I[i]/N*dt)
        I.append(I[i] + (beta*S[i]*I[i]/N - gamma*I[i])*dt)
        R.append(R[i] + gamma*I[i]*dt)
        t.append(t[i] + dt)
    
    return S, I, R, t


# Example usage

S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1

t_max = 100
dt = 0.1

S, I, R, t = simulate_SIR(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model Simulation')
plt.show()
