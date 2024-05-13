import numpy as np
import matplotlib.pyplot as plt

def SEIR_model(N, beta, gamma, sigma, I0, E0, R0, T):
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    dt = 1
    t = np.linspace(0, T, int(T/dt)+1)
    for _ in t[1:]:
        S.append(S[-1] - beta*S[-1]*I[-1]/N*dt)
        E.append(E[-1] + beta*S[-2]*I[-2]/N*dt - sigma*E[-1]*dt)
        I.append(I[-1] + sigma*E[-2]*dt - gamma*I[-1]*dt)
        R.append(R[-1] + gamma*I[-2]*dt)
    return S, E, I, R

N = 100000
beta = 0.2
gamma = 0.1
sigma = 0.1
I0 = 1
E0 = 0
R0 = 0
T = 100

S, E, I, R = SEIR_model(N, beta, gamma, sigma, I0, E0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
