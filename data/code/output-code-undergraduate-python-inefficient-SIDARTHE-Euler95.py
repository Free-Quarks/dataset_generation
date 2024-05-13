import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, I0, R0, D0, T, alpha, beta, gamma, delta, theta, epsilon):
    S0 = N - I0 - R0 - D0
    E0 = 0
    A0 = 0
    T0 = 0
    H0 = 0
    R = [R0]
    D = [D0]
    I = [I0]
    S = [S0]
    E = [E0]
    A = [A0]
    T = [T0]
    H = [H0]
    dt = T[1] - T[0]
    for i in range(1, len(T)):
        next_S = S[i-1] - (alpha * S[i-1] * (I[i-1] + theta * A[i-1])) * dt
        next_E = E[i-1] + (alpha * S[i-1] * (I[i-1] + theta * A[i-1]) - beta * E[i-1]) * dt
        next_A = A[i-1] + (epsilon * beta * E[i-1] - gamma * A[i-1]) * dt
        next_T = T[i-1] + (delta * (1 - epsilon) * beta * E[i-1] - delta * T[i-1]) * dt
        next_H = H[i-1] + (gamma * A[i-1] - delta * H[i-1]) * dt
        next_I = I[i-1] + (beta * E[i-1] - (gamma + delta) * I[i-1]) * dt
        next_R = R[i-1] + (delta * (1 - epsilon) * beta * E[i-1] + gamma * A[i-1] - delta * R[i-1]) * dt
        next_D = D[i-1] + (delta * T[i-1] + delta * (1 - epsilon) * beta * E[i-1] - delta * D[i-1]) * dt
        S.append(next_S)
        E.append(next_E)
        A.append(next_A)
        T.append(next_T)
        H.append(next_H)
        I.append(next_I)
        R.append(next_R)
        D.append(next_D)
    return S, E, A, T, H, I, R, D


N = 1000000
I0 = 1
R0 = 0
D0 = 0
T = np.linspace(0, 200, 200)

alpha = 0.2
beta = 1.75
gamma = 0.5
theta = 0.037
epsilon = 0.03
delta = 0.2

S, E, A, T, H, I, R, D = sidarthe_model(N, I0, R0, D0, T, alpha, beta, gamma, delta, theta, epsilon)

plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, A, label='Asymptomatic')
plt.plot(T, T, label='Tested')
plt.plot(T, H, label='Hospitalized')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.plot(T, D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
