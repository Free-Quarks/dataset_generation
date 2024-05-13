import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(beta, sigma, tau, p1, p2, p3, rho, tmax):
    # Initialize model parameters
    S = [999999]
    I = [1]
    D = [0]
    A = [0]
    R = [0]
    T = [0]
    H = [0]
    E = [0]

    dt = 0.01
    t = np.arange(0, tmax, dt)

    # Euler's method to solve ODE
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * (I[i-1] + p1 * A[i-1]) / (S[i-1] + I[i-1] + A[i-1] + R[i-1])
        dE = beta * S[i-1] * (I[i-1] + p1 * A[i-1]) / (S[i-1] + I[i-1] + A[i-1] + R[i-1]) - sigma * E[i-1]
        dI = sigma * (1 - tau) * E[i-1] - (p2 + p3 + rho) * I[i-1]
        dA = sigma * tau * E[i-1] - (p2 + p3) * A[i-1]
        dR = p2 * (I[i-1] + A[i-1])
        dT = p3 * I[i-1]
        dH = rho * I[i-1]

        S.append(S[i-1] + dt * dS)
        E.append(E[i-1] + dt * dE)
        I.append(I[i-1] + dt * dI)
        A.append(A[i-1] + dt * dA)
        R.append(R[i-1] + dt * dR)
        T.append(T[i-1] + dt * dT)
        H.append(H[i-1] + dt * dH)
        D.append(D[i-1] + dt * (dT + dH))

    return S, E, I, A, R, T, H, D


beta = 0.35
sigma = 1/5.2
tau = 0.6
p1 = 0.2
p2 = 0.5
p3 = 0.3
rho = 0.05
tmax = 100

S, E, I, A, R, T, H, D = sidarthe_model(beta, sigma, tau, p1, p2, p3, rho, tmax)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Transferred')
plt.plot(H, label='Hospitalized')
plt.plot(D, label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
