import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(s0, i0, r0, beta, gamma, t0, tmax, dt):
    t = np.arange(t0, tmax+dt, dt)
    N = s0 + i0 + r0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = s0
    I[0] = i0
    R[0] = r0
    
    def dSdt(t, S, I, beta):
        return -beta * S * I / N
    
    def dIdt(t, S, I, beta, gamma):
        return beta * S * I / N - gamma * I
    
    def dRdt(t, I, gamma):
        return gamma * I
    
    for i in range(len(t)-1):
        k1 = dt * dSdt(t[i], S[i], I[i], beta)
        l1 = dt * dIdt(t[i], S[i], I[i], beta, gamma)
        m1 = dt * dRdt(t[i], I[i], gamma)
        
        k2 = dt * dSdt(t[i]+dt/2, S[i]+k1/2, I[i]+l1/2, beta)
        l2 = dt * dIdt(t[i]+dt/2, S[i]+k1/2, I[i]+l1/2, beta, gamma)
        m2 = dt * dRdt(t[i]+dt/2, I[i]+l1/2, gamma)
        
        k3 = dt * dSdt(t[i]+dt/2, S[i]+k2/2, I[i]+l2/2, beta)
        l3 = dt * dIdt(t[i]+dt/2, S[i]+k2/2, I[i]+l2/2, beta, gamma)
        m3 = dt * dRdt(t[i]+dt/2, I[i]+l2/2, gamma)
        
        k4 = dt * dSdt(t[i]+dt, S[i]+k3, I[i]+l3, beta)
        l4 = dt * dIdt(t[i]+dt, S[i]+k3, I[i]+l3, beta, gamma)
        m4 = dt * dRdt(t[i]+dt, I[i]+l3, gamma)
        
        S[i+1] = S[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        I[i+1] = I[i] + (1/6) * (l1 + 2*l2 + 2*l3 + l4)
        R[i+1] = R[i] + (1/6) * (m1 + 2*m2 + 2*m3 + m4)
        
    return S, I, R


# Example usage
s0 = 999
i0 = 1
r0 = 0
beta = 0.3
gamma = 0.1
t0 = 0
tmax = 100
dt = 0.1

S, I, R = SIR_RK4(s0, i0, r0, beta, gamma, t0, tmax, dt)

# Plotting
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()
