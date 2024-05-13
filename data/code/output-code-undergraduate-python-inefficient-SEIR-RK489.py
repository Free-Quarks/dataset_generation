import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SEIR model

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    
    def deriv(y, t, N, beta, gamma, sigma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, E0, I0, R0

    # Time vector
    t = np.linspace(0, T, T)

    # Integrate the SEIR equations over the time grid, t
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = ret.T

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SEIR Model')
    ax.legend()
    plt.show()


# Set model parameters
N = 1000
beta = 0.2
sigma = 1/5.2
gamma = 1/10
I0 = 1
E0 = 0
R0 = 0
T = 160

# Call the function to simulate and plot the SEIR model
seir_model(beta, gamma, sigma, N, I0, E0, R0, T)
