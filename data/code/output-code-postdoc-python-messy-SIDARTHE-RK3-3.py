import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, I0, R0, D0, T, beta, alpha, gamma, rho, theta, delta):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Initial number of deaths, D0.
    # Contact rate, beta.
    # Mean recovery rate, gamma.
    # Mean hospitalization rate, alpha.
    # Mean intensive care rate, rho.
    # Mean invasive mechanical ventilation rate, theta.
    #Mean death rate, delta.
    
    S0 = N - I0 - R0 - D0
    # Everyone else, S0, is susceptible to infection initially.
    
    S = [S0]
    I = [I0]
    D = [D0]
    A = [I0 - R0 - D0]
    R = [R0]
    T = np.linspace(0, T, T)
    
    # Euler method
    dt = T[1] - T[0]
    for t in range(1,T):
        dSdt = -beta * S[t-1] * (I[t-1] + A[t-1]) / N
        dIdt = beta * S[t-1] * (I[t-1] + A[t-1]) / N - (alpha + gamma + rho + theta + delta) * I[t-1]
        dDdt = delta * I[t-1]
        dAdt = alpha * I[t-1] - R[t-1]
        dRdt = gamma * I[t-1] + rho * R[t-1] + theta * R[t-1]
        
        S.append(S[t-1] + dt * dSdt)
        I.append(I[t-1] + dt * dIdt)
        D.append(D[t-1] + dt * dDdt)
        A.append(A[t-1] + dt * dAdt)
        R.append(R[t-1] + dt * dRdt)
    
    return S, I, D, A, R


N = 1000000
I0, R0, D0 = 1, 0, 0
T = 100
beta, alpha, gamma, rho, theta, delta = 0.2, 0.1, 0.05, 0.02, 0.01, 0.01

S, I, D, A, R = sidarthe_model(N, I0, R0, D0, T, beta, alpha, gamma, rho, theta, delta)

plt.plot(T, S, label='Susceptible')
plt.plot(T, I, label='Infected')
plt.plot(T, D, label='Deaths')
plt.plot(T, A, label='Asymptomatic')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIDARTHE Compartmental Model')
plt.legend()
plt.show()
