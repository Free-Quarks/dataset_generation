import numpy as np
import matplotlib.pyplot as plt

def serid_model(beta, gamma, mu, N, I0, R0, D0, T):
    # Define the initial conditions
    S0 = N - I0 - R0 - D0
    Y0 = [S0, I0, R0, D0]
    
    # Define the differential equations
    def deriv(Y, t, beta, gamma, mu, N):
        S, I, R, D = Y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - (gamma + mu) * I
        dRdt = gamma * I
        dDdt = mu * I
        return dSdt, dIdt, dRdt, dDdt
    
    # Integrate the differential equations over time
    t = np.linspace(0, T, T+1)
    Y = np.zeros((T+1, 4))
    Y[0] = Y0
    for i in range(T):
        dYdt = deriv(Y[i], t[i], beta, gamma, mu, N)
        Y[i+1] = Y[i] + dYdt
    
    # Return the results
    return t, Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]

# Set the model parameters
beta = 0.2
gamma = 0.1
mu = 0.01
N = 1000
I0 = 1
R0 = 0
D0 = 0
T = 100

# Call the model function
t, S, I, R, D = serid_model(beta, gamma, mu, N, I0, R0, D0, T)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model')
plt.legend()
plt.show()
