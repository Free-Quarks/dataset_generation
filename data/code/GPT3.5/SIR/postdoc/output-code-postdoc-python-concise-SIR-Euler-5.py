import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    t = np.linspace(0, t_end, int(t_end/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for i in range(len(t)-1):
        S[i+1] = S[i] - beta*S[i]*I[i]*dt/N
        I[i+1] = I[i] + (beta*S[i]*I[i]/N - gamma*I[i])*dt
        R[i+1] = R[i] + gamma*I[i]*dt

    return S, I, R


beta = 0.3
# infection rate

gamma = 0.1
# recovery rate

N = 1000
# total population

I0 = 1
# initial infected individuals

R0 = 0
# initial recovered individuals

t_end = 100
# simulation time

dt = 0.1
# time step

S, I, R = SIR_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

