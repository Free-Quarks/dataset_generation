import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    def deriv(y, t, N, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    y0 = N - I0 - R0, E0, I0, R0
    t = np.arange(0, T+1)
    
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T
    
    return S, E, I, R

# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.3
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

# Run the model
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
