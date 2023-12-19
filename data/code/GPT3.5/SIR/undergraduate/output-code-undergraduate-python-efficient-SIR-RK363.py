import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T):
    
    def f(t, u):
        S, I, R = u
        return [-beta*S*I, beta*S*I - gamma*I, gamma*I]
    
    t = np.linspace(0, T, T+1)
    dt = t[1] - t[0]
    
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    
    I[0] = I0
    S[0] = N - I0
    R[0] = 0
    
    for i in range(T):
        u = [S[i], I[i], R[i]]
        k1 = f(t[i], u)
        k2 = f(t[i] + 0.5*dt, [u[j] + 0.5*dt*k1[j] for j in range(len(u))])
        k3 = f(t[i] + dt, [u[j] - dt*k1[j] + 2*dt*k2[j] for j in range(len(u))])
        S[i+1] = S[i] - (dt/6)*(k1[0] + 4*k2[0] + k3[0])
        I[i+1] = I[i] - (dt/6)*(k1[1] + 4*k2[1] + k3[1])
        R[i+1] = R[i] - (dt/6)*(k1[2] + 4*k2[2] + k3[2])
    
    return S, I, R


N = 1000
I0 = 1
beta = 0.2
gamma = 0.1
T = 100

S, I, R = SIR_RK3(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
