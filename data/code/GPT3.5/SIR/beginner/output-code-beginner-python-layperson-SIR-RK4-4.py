import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    t = np.arange(0, t_max+dt, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1 = -beta*S[i-1]*I[i-1]/N
        k2 = -beta*(S[i-1]+0.5*k1*dt)*(I[i-1]+0.5*k1*dt)/N
        k3 = -beta*(S[i-1]+0.5*k2*dt)*(I[i-1]+0.5*k2*dt)/N
        k4 = -beta*(S[i-1]+k3*dt)*(I[i-1]+k3*dt)/N
        S[i] = S[i-1] + (k1+2*k2+2*k3+k4)*dt
        k1 = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
        k2 = beta*S[i-1]*(I[i-1]+0.5*k1*dt)/N - gamma*(I[i-1]+0.5*k1*dt)
        k3 = beta*S[i-1]*(I[i-1]+0.5*k2*dt)/N - gamma*(I[i-1]+0.5*k2*dt)
        k4 = beta*S[i-1]*(I[i-1]+k3*dt)/N - gamma*(I[i-1]+k3*dt)
        I[i] = I[i-1] + (k1+2*k2+2*k3+k4)*dt
        k1 = gamma*I[i-1]
        k2 = gamma*(I[i-1]+0.5*k1*dt)
        k3 = gamma*(I[i-1]+0.5*k2*dt)
        k4 = gamma*(I[i-1]+k3*dt)
        R[i] = R[i-1] + (k1+2*k2+2*k3+k4)*dt
    return S, I, R

beta = 0.3
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t_max = 100
dt = 0.1
S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
