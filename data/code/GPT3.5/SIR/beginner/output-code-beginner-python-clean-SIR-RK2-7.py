import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, days):
    N = S0 + I0 + R0
    S = [S0]
    I = [I0]
    R = [R0]

    dt = 0.01
    t = np.linspace(0, days, int(days/dt) + 1)

    for i in range(1, len(t)):
        dS = -beta*S[i-1]*I[i-1]/N
dI = beta*S[i-1]*I[i-1]/N - gamma*I[i-1]
dR = gamma*I[i-1]

        S.append(S[i-1] + dt*dS)
        I.append(I[i-1] + dt*dI)
        R.append(R[i-1] + dt*dR)

    return t, S, I, R


beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
days = 100

t, S, I, R = SIR_model(beta, gamma, S0, I0, R0, days)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
