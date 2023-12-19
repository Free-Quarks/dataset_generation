import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, T):
    
    N = S0 + I0 + R0  # Total population size
    
    S = [S0]  # List to store the number of susceptible individuals at each time step
    I = [I0]  # List to store the number of infected individuals at each time step
    R = [R0]  # List to store the number of recovered individuals at each time step
    
    dt = 0.1  # Time step size
    t = np.arange(0, T+dt, dt)  # Time array
    
    for _ in range(len(t)-1):
        
        dS = -beta * S[-1] * I[-1] / N  # Change in susceptible individuals
        dI = (beta * S[-1] * I[-1] / N) - gamma * I[-1]  # Change in infected individuals
        dR = gamma * I[-1]  # Change in recovered individuals
        
        S.append(S[-1] + dt * dS)  # Update susceptible individuals
        I.append(I[-1] + dt * dI)  # Update infected individuals
        R.append(R[-1] + dt * dR)  # Update recovered individuals
    
    return S, I, R


# Example usage
beta = 0.3
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
T = 50

S, I, R = SIR_model(beta, gamma, S0, I0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
