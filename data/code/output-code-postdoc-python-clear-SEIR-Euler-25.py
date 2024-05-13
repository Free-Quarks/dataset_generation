import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, delta, I0, R0, T):
    # Initial conditions
    S0 = N - I0 - R0
    E0 = 0
    
    # Compartments
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    
    # Time steps
    dt = 0.1
    t = np.arange(0, T+dt, dt)
    
    # Euler method
    for i in range(1, len(t)):
        new_S = S[i-1] - (beta * S[i-1] * I[i-1] / N) * dt
        new_E = E[i-1] + (beta * S[i-1] * I[i-1] / N - delta * E[i-1]) * dt
        new_I = I[i-1] + (delta * E[i-1] - gamma * I[i-1]) * dt
        new_R = R[i-1] + (gamma * I[i-1]) * dt
        
        S.append(new_S)
        E.append(new_E)
        I.append(new_I)
        R.append(new_R)
    
    return S, E, I, R


# Example usage
N = 1000
beta = 0.3
gamma = 0.1
delta = 0.2
I0 = 1
R0 = 0
T = 100

S, E, I, R = seir_model(N, beta, gamma, delta, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.show()
