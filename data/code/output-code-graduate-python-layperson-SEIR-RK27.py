import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Set initial conditions
    S0 = N - I0 - E0 - R0
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    
    dt = T[1] - T[0]
    
    # Run the model
    for t in range(1, len(T)):
        S_to_E = beta * S[t-1] * I[t-1] / N
        E_to_I = sigma * E[t-1]
        I_to_R = gamma * I[t-1]
        
        S_new = S[t-1] - dt * S_to_E
        E_new = E[t-1] + dt * (S_to_E - E_to_I)
        I_new = I[t-1] + dt * (E_to_I - I_to_R)
        R_new = R[t-1] + dt * I_to_R
        
        S.append(S_new)
        E.append(E_new)
        I.append(I_new)
        R.append(R_new)
    
    return S, E, I, R

# Example usage
beta = 0.5
gamma = 0.1
sigma = 0.2
N = 1000
I0 = 10
E0 = 5
R0 = 0
T = np.linspace(0, 100, 1000)
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
