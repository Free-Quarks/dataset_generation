import numpy as np
import matplotlib.pyplot as plt

def seird_model(N, I0, R0, E0, D0, beta, gamma, sigma, mu, num_days):
    
    S0 = N - I0 - R0 - E0 - D0
    
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    D = [D0]
    
    dt = 0.01
    t = np.arange(0, num_days, dt)
    
    for _ in t:
        
        dSdt = -beta * S[-1] * I[-1] / N
        dEdt = beta * S[-1] * I[-1] / N - sigma * E[-1]
        dIdt = sigma * E[-1] - gamma * I[-1] - mu * I[-1]
        dRdt = gamma * I[-1]
        dDdt = mu * I[-1]
        
        S.append(S[-1] + dSdt * dt)
        E.append(E[-1] + dEdt * dt)
        I.append(I[-1] + dIdt * dt)
        R.append(R[-1] + dRdt * dt)
        D.append(D[-1] + dDdt * dt)
    
    return S, E, I, R, D


N = 100000
I0 = 1
R0 = 0
E0 = 0
D0 = 0
beta = 0.2
gamma = 0.1
sigma = 0.05
mu = 0.01
num_days = 100

S, E, I, R, D = seird_model(N, I0, R0, E0, D0, beta, gamma, sigma, mu, num_days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIRD Model')
plt.legend()
plt.show()
