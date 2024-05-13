import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    dt = 0.1
    t = np.linspace(0, T, int(T/dt) + 1)
    S = np.zeros(t.shape)
    E = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        dE = beta*S[i-1]*I[i-1]/N - sigma*E[i-1]
        dI = sigma*E[i-1] - gamma*I[i-1]
        dR = gamma*I[i-1]
        
        S[i] = S[i-1] - dt*beta*S[i-1]*I[i-1]/N
        E[i] = E[i-1] + dt*(dE)
        I[i] = I[i-1] + dt*(dI)
        R[i] = R[i-1] + dt*(dR)
        
    return t, S, E, I, R

# Example usage
T = 100
N = 1000
I0, E0, R0 = 1, 0, 0
beta, gamma, sigma = 0.2, 0.1, 0.05

t, S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
