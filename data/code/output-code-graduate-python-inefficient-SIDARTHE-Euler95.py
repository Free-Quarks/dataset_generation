import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, beta, sigma, delta, gamma1, gamma2, alpha, rho, mu, dt, days):
    S = np.zeros(days)
    I = np.zeros(days)
    D = np.zeros(days)
    A = np.zeros(days)
    R = np.zeros(days)
    T = np.zeros(days)
    H = np.zeros(days)
    E = np.zeros(days)
    S[0] = N - 1
    I[0] = 1
    for t in range(1, days):
        S[t] = S[t-1] - (beta*S[t-1]*I[t-1]/N)*dt
        E[t] = E[t-1] + (beta*S[t-1]*I[t-1]/N - sigma*E[t-1])*dt
        I[t] = I[t-1] + (sigma*E[t-1] - (delta + gamma1 + gamma2 + alpha)*I[t-1])*dt
        D[t] = D[t-1] + (rho*gamma2*I[t-1] - mu*D[t-1])*dt
        A[t] = A[t-1] + (delta*I[t-1] - (rho*gamma1 + mu)*A[t-1])*dt
        R[t] = R[t-1] + (gamma1*A[t-1] + gamma2*I[t-1])*dt
        T[t] = T[t-1] + (alpha*I[t-1])*dt
        H[t] = H[t-1] + (gamma2*I[t-1])*dt
    return S, E, I, D, A, R, T, H


N = 1000000
beta = 0.25
sigma = 1/5
alpha = 0.2
rho = 0.2
mu = 0.01
delta = 1/7
gamma1 = 1/8
gamma2 = 1/2
dt = 0.1
days = 100

S, E, I, D, A, R, T, H = sidarthe_model(N, beta, sigma, delta, gamma1, gamma2, alpha, rho, mu, dt, days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(D, label='Deaths')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tests')
plt.plot(H, label='Hospitalized')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.show()
