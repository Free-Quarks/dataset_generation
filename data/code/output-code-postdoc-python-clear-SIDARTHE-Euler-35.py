import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, t_max, beta, theta, delta, gamma, alpha, rho):
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    
    S = [S0]
    I = [I0]
    D = [D0]
    A = [A0]
    R = [R0]
    T = [T0]
    H = [H0]
    E = [E0]
    
    dt = 0.1
    t = np.arange(0, t_max, dt)
    
    for _ in t[1:]:
        dS = -beta * S[-1] * (I[-1] + theta * A[-1]) / N
        dI = beta * S[-1] * (I[-1] + theta * A[-1]) / N - (delta + gamma + alpha) * I[-1]
        dD = delta * I[-1]
        dA = alpha * I[-1] - rho * A[-1]
        dR = gamma * I[-1] + rho * A[-1]
        dT = theta * A[-1]
        dH = delta * I[-1]
        dE = (1 - delta) * I[-1]
        
        S.append(S[-1] + dS * dt)
        I.append(I[-1] + dI * dt)
        D.append(D[-1] + dD * dt)
        A.append(A[-1] + dA * dt)
        R.append(R[-1] + dR * dt)
        T.append(T[-1] + dT * dt)
        H.append(H[-1] + dH * dt)
        E.append(E[-1] + dE * dt)
    
    return S, I, D, A, R, T, H, E


N = 1000000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
E0 = 0

beta = 0.4
theta = 0.2
alpha = 0.1
rho = 0.03

delta = 0.02

gamma = 0.1


S, I, D, A, R, T, H, E = sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, t_max=100, beta=beta, theta=theta, delta=delta, gamma=gamma, alpha=alpha, rho=rho)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()
