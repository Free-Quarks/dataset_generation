import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def sidarthe(y, t, beta, sigma, gamma, tau, mu, alpha, rho):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dS = -beta * S * (I + alpha * A) / N
    dI = (beta * S * (I + alpha * A) / N) - (sigma * I) - (gamma * I) - (rho * I)
    dD = mu * sigma * I
    dA = sigma * I - (tau * A) - (gamma * alpha * A) - (rho * alpha * A)
    dR = gamma * I + gamma * alpha * A
    dT = tau * A
    dH = rho * I + rho * alpha * A
    dE = rho * I
    return [dS, dI, dD, dA, dR, dT, dH, dE]


def run_sidarthe(S, I, D, A, R, T, H, E, beta, sigma, gamma, tau, mu, alpha, rho, t_max):
    y0 = [S, I, D, A, R, T, H, E]
    t = np.linspace(0, t_max, t_max+1)
    ret = odeint(sidarthe, y0, t, args=(beta, sigma, gamma, tau, mu, alpha, rho))
    S, I, D, A, R, T, H, E = ret.T
    return S, I, D, A, R, T, H, E


# Example usage
S, I, D, A, R, T, H, E = run_sidarthe(S=1000, I=10, D=0, A=0, R=0, T=0, H=0, E=0, beta=0.2, sigma=0.1, gamma=0.02, tau=0.05, mu=0.01, alpha=0.5, rho=0.1, t_max=100)
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Dead')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Tested')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='Exposed')
plt.legend()
plt.show()
