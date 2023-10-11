import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, I0, R0, M0, T, dt):
    # Initial conditions
    S0 = N - I0 - R0 - M0
    E0 = 0
    A0 = 0
    D0 = 0
    H0 = 0
    T0 = 0
    
    # Parameters
    alpha = 0.2
    beta = 0.2
    gamma = 0.15
    delta = 0.02
    mu = 0.05
    
    S = [S0]
    E = [E0]
    A = [A0]
    D = [D0]
    H = [H0]
    T = [T0]
    I = [I0]
    R = [R0]
    M = [M0]
    
    for t in range(1, T+1):
        DS = -beta * S[t-1] * I[t-1] / N
        DE = beta * S[t-1] * I[t-1] / N - alpha * E[t-1]
        DA = alpha * (1 - delta) * E[t-1] - gamma * A[t-1]
        DD = alpha * delta * E[t-1] - mu * D[t-1]
        DH = gamma * A[t-1] - mu * H[t-1]
        DI = beta * S[t-1] * I[t-1] / N - (alpha + mu) * I[t-1]
        DR = gamma * A[t-1] + (alpha + mu) * I[t-1]
        DM = mu * (D[t-1] + H[t-1])
        
        S.append(S[t-1] + dt * DS)
        E.append(E[t-1] + dt * DE)
        A.append(A[t-1] + dt * DA)
        D.append(D[t-1] + dt * DD)
        H.append(H[t-1] + dt * DH)
        T.append(T[t-1] + dt)
        I.append(I[t-1] + dt * DI)
        R.append(R[t-1] + dt * DR)
        M.append(M[t-1] + dt * DM)
    
    return S, E, A, D, H, T, I, R, M


# Example usage
N = 1000000
I0 = 100
R0 = 0
M0 = 10
T = 100
dt = 0.1

S, E, A, D, H, T, I, R, M = sidarthe_model(N, I0, R0, M0, T, dt)

plt.plot(T, S, label='S')
plt.plot(T, E, label='E')
plt.plot(T, A, label='A')
plt.plot(T, D, label='D')
plt.plot(T, H, label='H')
plt.plot(T, I, label='I')
plt.plot(T, R, label='R')
plt.plot(T, M, label='M')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()
