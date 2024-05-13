import numpy as np
import matplotlib.pyplot as plt
def seir_model(N, beta, gamma, delta, days):
    S = [N-1]
    E = [1]
    I = [0]
    R = [0]
    dt = 0.1
    t = np.arange(0, days, dt)
    for _ in range(len(t)-1):
        dS = -beta*S[-1]*I[-1]/N
        dE = beta*S[-1]*I[-1]/N - delta*E[-1]
        dI = delta*E[-1] - gamma*I[-1]
        dR = gamma*I[-1]
        S.append(S[-1] + dS*dt)
        E.append(E[-1] + dE*dt)
        I.append(I[-1] + dI*dt)
        R.append(R[-1] + dR*dt)
    return S, E, I, R

N = 1000
beta = 0.4
gamma = 0.1
delta = 0.2
days = 100

S, E, I, R = seir_model(N, beta, gamma, delta, days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SEIR Model Simulation')
plt.show()
