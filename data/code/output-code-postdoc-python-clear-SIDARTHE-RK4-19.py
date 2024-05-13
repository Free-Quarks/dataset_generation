import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, gamma, delta, theta, alpha, rho, N, tmax, S0, I0, D0, A0, R0, T0, H0, E0):
    def deriv(y, t, beta, gamma, delta, theta, alpha, rho, N):
        S, I, D, A, R, T, H, E = y
        dSdt = -beta * S * (I + delta * A) / N
        dIdt = beta * S * (I + delta * A) / N - (gamma + alpha + rho) * I
        dDdt = theta * alpha * I - (rho + delta) * D
        dAdt = (1 - theta) * alpha * I - (gamma + delta) * A
        dRdt = gamma * (I + A + T + H)
        dTdt = rho * (I + A + T + H) - (delta + gamma) * T
        dHdt = delta * (I + A + T + H)
        dEdt = rho * T - (gamma + alpha + delta) * E
        return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt
    
    t = np.linspace(0, tmax, tmax)
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    
    ret = odeint(deriv, y0, t, args=(beta, gamma, delta, theta, alpha, rho, N))
    S, I, D, A, R, T, H, E = ret.T
    
    return S, I, D, A, R, T, H, E


# example usage
beta = 0.2
gamma = 0.1
theta = 0.8
alpha = 0.3
rho = 0.05
N = 1000
tmax = 100
S0, I0, D0, A0, R0, T0, H0, E0 = N-1, 1, 0, 0, 0, 0, 0, 0

S, I, D, A, R, T, H, E = sidarthe_model(beta, gamma, delta, theta, alpha, rho, N, tmax, S0, I0, D0, A0, R0, T0, H0, E0)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Active')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.legend()
plt.show()

