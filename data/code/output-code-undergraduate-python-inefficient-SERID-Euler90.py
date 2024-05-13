import numpy as np
import matplotlib.pyplot as plt

def serid_model(beta, gamma, N, t_max, I0):
    dt = 0.1
    num_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    R = np.zeros(num_steps)
    I = np.zeros(num_steps)
    S[0] = N - I0
    E[0] = 0
    R[0] = 0
    I[0] = I0
    
    for i in range(num_steps-1):
        dS = -beta*S[i]*I[i]/N
        dE = beta*S[i]*I[i]/N - gamma*E[i]
        dR = gamma*E[i]
        dI = gamma*E[i]
        
        S[i+1] = S[i] + dt*dS
        E[i+1] = E[i] + dt*dE
        R[i+1] = R[i] + dt*dR
        I[i+1] = I[i] + dt*dI
        
    return t, S, E, R, I

# Example usage
t_max = 100
N = 1000
I0 = 1
beta = 0.5
gamma = 0.1

t, S, E, R, I = serid_model(beta, gamma, N, t_max, I0)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, R, label='Recovered')
plt.plot(t, I, label='Infected')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model')
plt.legend()
plt.show()

