import numpy as np
import matplotlib.pyplot as plt

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(N, E, I, R, beta, gamma, sigma, days):
    # Initial conditions
    S = N - E - I - R
    y0 = S, E, I, R
    
    # Vector of time points
    t = np.linspace(0, days, days)
    
    # Integrate the SEIR equations over the time grid
    res = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = res.T
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'm', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SEIR Model')
    plt.grid(True)
    plt.show()


N = 1000  # Total population
E = 10    # Initial number of exposed individuals
I = 1     # Initial number of infected individuals
R = 0     # Initial number of recovered individuals
beta = 0.2  # Contact rate
gamma = 0.1  # Recovery rate
sigma = 0.3  # Incubation rate

run_seir_model(N, E, I, R, beta, gamma, sigma, days=180)
