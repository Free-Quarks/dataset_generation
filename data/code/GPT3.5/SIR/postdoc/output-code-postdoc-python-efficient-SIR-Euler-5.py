import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, T):
    # Total population
    S0 = N - I0 - R0
    
    # Initial conditions
    S = [S0]
    I = [I0]
    R = [R0]
    
    # Time vector
    t = np.linspace(0, T, T+1)
    
    # Euler's method
    dt = 1
    for _ in range(T):
        S.append(S[-1] - beta*S[-1]*I[-1]*dt/N)
        I.append(I[-1] + (beta*S[-2]*I[-1]/N - gamma*I[-1])*dt)
        R.append(R[-1] + gamma*I[-1]*dt)
    
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
T = 100

# Run model
S, I, R = SIR_model(beta, gamma, N, I0, R0, T)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
