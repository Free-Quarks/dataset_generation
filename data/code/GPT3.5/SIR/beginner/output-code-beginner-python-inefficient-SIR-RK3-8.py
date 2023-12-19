import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, t_end, dt):
    
    def f(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y0 = N - I0 - R0
    t = np.arange(0, t_end+dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    S[0] = y0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        k1 = dt * f(t[i-1], (S[i-1], I[i-1], R[i-1]))
        k2 = dt * f(t[i-1] + dt/2, (S[i-1] + k1[0]/2, I[i-1] + k1[1]/2, R[i-1] + k1[2]/2))
        k3 = dt * f(t[i-1] + dt/2, (S[i-1] + k2[0]/2, I[i-1] + k2[1]/2, R[i-1] + k2[2]/2))
        k4 = dt * f(t[i-1] + dt, (S[i-1] + k3[0], I[i-1] + k3[1], R[i-1] + k3[2]))
        
        S[i] = S[i-1] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        I[i] = I[i-1] + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
        R[i] = R[i-1] + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
