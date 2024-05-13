import numpy as np
import matplotlib.pyplot as plt

def seird_model(beta, sigma, gamma, delta, N, I0, E0, R0, D0, days):
    S = N - I0 - E0 - R0 - D0
    y0 = [S, E0, I0, R0, D0]
    t = np.arange(days)
    
    def seird(y, t, beta, sigma, gamma, delta, N):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - delta * I
        dRdt = gamma * I
        dDdt = delta * I
        return dSdt, dEdt, dIdt, dRdt, dDdt
    
    result = odeint(seird, y0, t, args=(beta, sigma, gamma, delta, N))
    S, E, I, R, D = result.T
    
    return t, S, E, I, R, D

# Example usage
beta = 1.75
sigma = 1/5.2
gamma = 1/2.9
delta = 0.02
N = 100000
I0 = 10
E0 = 5
R0 = 0
D0 = 0
days = 100

t, S, E, I, R, D = seird_model(beta, sigma, gamma, delta, N, I0, E0, R0, D0, days)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('SEIRD Model Simulation')
plt.show()
