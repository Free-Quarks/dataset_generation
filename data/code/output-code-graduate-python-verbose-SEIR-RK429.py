import numpy as np
import matplotlib.pyplot as plt

def SEIR_RK4(beta, sigma, gamma, N, I0, E0, R0, t_max, dt):
    def derivs(y, t, beta, sigma, gamma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, t_max, int(t_max/dt))
    y0 = N - I0 - E0 - R0
    y = S0, E0, I0, R0

    S, E, I, R = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    S[0], E[0], I[0], R[0] = y

    for i in range(1, len(t)):
        s1, e1, i1, r1 = derivs(y, t[i-1], beta, sigma, gamma, N)
        s2, e2, i2, r2 = derivs(y + dt/2 * s1, t[i-1] + dt/2, beta, sigma, gamma, N)
        s3, e3, i3, r3 = derivs(y + dt/2 * s2, t[i-1] + dt/2, beta, sigma, gamma, N)
        s4, e4, i4, r4 = derivs(y + dt * s3, t[i-1] + dt, beta, sigma, gamma, N)
        y += dt/6 * (s1 + 2*s2 + 2*s3 + s4), dt/6 * (e1 + 2*e2 + 2*e3 + e4), dt/6 * (i1 + 2*i2 + 2*i3 + i4), dt/6 * (r1 + 2*r2 + 2*r3 + r4)
        S[i], E[i], I[i], R[i] = y

    return S, E, I, R

# Example usage
beta = 0.5
sigma = 0.1
gamma = 0.05
N = 1000
I0 = 10
E0 = 5
R0 = 0

t_max = 100
dt = 0.1

S, E, I, R = SEIR_RK4(beta, sigma, gamma, N, I0, E0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model using RK4')
plt.show()
