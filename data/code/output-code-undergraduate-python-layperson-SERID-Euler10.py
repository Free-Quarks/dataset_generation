import numpy as np
import matplotlib.pyplot as plt

def SERID(N, beta, gamma, delta, mu, sigma, I0, E0, R0, D0, T):
    S0 = N - I0 - E0 - R0 - D0
    S = [S0]
    E = [E0]
    R = [R0]
    I = [I0]
    D = [D0]

    dt = 1
    t = np.linspace(0, T, int(T/dt) + 1)

    for _ in t[1:]:
        dS = -beta*S[-1]*I[-1]/N
        dE = beta*S[-1]*I[-1]/N - sigma*E[-1] - delta*E[-1]
        dI = sigma*E[-1] - gamma*I[-1] - mu*I[-1]
        dR = gamma*I[-1]
        dD = delta*E[-1] + mu*I[-1]

        S.append(S[-1] + dS*dt)
        E.append(E[-1] + dE*dt)
        I.append(I[-1] + dI*dt)
        R.append(R[-1] + dR*dt)
        D.append(D[-1] + dD*dt)

    return S, E, I, R, D


N = 1000000
beta = 0.2
sigma = 1/5
gamma = 1/7
mu = 0.01
delta = 0.01
I0 = 100
E0 = 100
R0 = 0
D0 = 0
T = 365

S, E, I, R, D = SERID(N, beta, gamma, delta, mu, sigma, I0, E0, R0, D0, T)

plt.figure(figsize=(10,6))
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
