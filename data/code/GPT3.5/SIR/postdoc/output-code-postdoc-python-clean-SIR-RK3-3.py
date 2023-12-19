import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, T, dt):
    # Initial conditions
    S0 = N - I0
    R0 = 0
    
    # Time vector
    t = np.arange(0, T+dt, dt)
    
    # Arrays to store the values
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 3rd order algorithm
    for i in range(1, len(t)):
        k1 = -beta*S[i-1]*I[i-1]/N
        k2 = -beta*(S[i-1] + 0.5*dt*k1)*(I[i-1] + 0.5*dt*k1)/N
        k3 = -beta*(S[i-1] - dt*k1 + 2*dt*k2)*(I[i-1] - dt*k1 + 2*dt*k2)/N
        
        m1 = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
        m2 = beta*(S[i-1] + 0.5*dt*k1)*(I[i-1] + 0.5*dt*k1)/N - gamma*(I[i-1] + 0.5*dt*m1)
        m3 = beta*(S[i-1] - dt*k1 + 2*dt*k2)*(I[i-1] - dt*k1 + 2*dt*k2)/N - gamma*(I[i-1] - dt*m1 + 2*dt*m2)
        
        S[i] = S[i-1] + dt*(k1 + 4*k2 + k3)/6
        I[i] = I[i-1] + dt*(m1 + 4*m2 + m3)/6
        R[i] = R[i-1] + dt*(gamma*I[i-1] + gamma*(I[i-1] + dt*m1) + gamma*(I[i-1] - dt*m1 + 2*dt*m2))/6
    
    return S, I, R


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, N, I0, T, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
