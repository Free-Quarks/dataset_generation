import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, t_max):
    dt = 0.1
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(len(t) - 1):
        dS_dt = -beta * S[i] * I[i] / N
        dI_dt = beta * S[i] * I[i] / N - gamma * I[i]
        dR_dt = gamma * I[i]
        
        k1_S = dt * dS_dt
        k1_I = dt * dI_dt
        k1_R = dt * dR_dt
        
        k2_S = dt * (-beta * (S[i] + k1_S/2) * (I[i] + k1_I/2) / N)
        k2_I = dt * (beta * (S[i] + k1_S/2) * (I[i] + k1_I/2) / N - gamma * (I[i] + k1_I/2))
        k2_R = dt * (gamma * (I[i] + k1_I/2))
        
        S[i+1] = S[i] + k2_S
        I[i+1] = I[i] + k2_I
        R[i+1] = R[i] + k2_R
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
t_max = 100

S, I, R = sir_model(beta, gamma, N, I0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
