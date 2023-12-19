import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_max, h):
    
    def f_SIR(S, I, R):
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        return dS, dI, dR
    
    t = np.arange(0, t_max+h, h)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        k1 = h * np.array(f_SIR(S[i-1], I[i-1], R[i-1]))
        k2 = h * np.array(f_SIR(S[i-1] + 0.5 * k1[0], I[i-1] + 0.5 * k1[1], R[i-1] + 0.5 * k1[2]))
        k3 = h * np.array(f_SIR(S[i-1] - k1[0] + 2 * k2[0], I[i-1] - k1[1] + 2 * k2[1], R[i-1] - k1[2] + 2 * k2[2]))
        
        S[i] = S[i-1] + (1/6) * (k1[0] + 4 * k2[0] + k3[0])
        I[i] = I[i-1] + (1/6) * (k1[1] + 4 * k2[1] + k3[1])
        R[i] = R[i-1] + (1/6) * (k1[2] + 4 * k2[2] + k3[2])
    
    return S, I, R

beta = 0.3
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
t_max = 100
h = 0.1

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_max, h)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
