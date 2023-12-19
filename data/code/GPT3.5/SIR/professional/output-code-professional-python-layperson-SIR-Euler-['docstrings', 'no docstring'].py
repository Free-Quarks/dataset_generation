import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, N, I0, R0, days):
    S0 = N - I0 - R0
    S = [S0]
    I = [I0]
    R = [R0]
    dt = 0.1
    t = np.linspace(0,days,int(days/dt)+1)
    for _ in t[1:]:
        dSdt = -beta*S[-1]*I[-1]/N
        dIdt = beta*S[-1]*I[-1]/N - gamma*I[-1]
        dRdt = gamma*I[-1]
        S.append(S[-1] + dt*dSdt)
        I.append(I[-1] + dt*dIdt)
        R.append(R[-1] + dt*dRdt)
    return S, I, R

N = 1000
I0, R0 = 10, 0
beta, gamma = 0.2, 0.1

S, I, R = sir_model(beta, gamma, N, I0, R0, 100)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
