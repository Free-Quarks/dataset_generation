import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, population, initial_infected, total_days):
    # Initialize arrays
    t = np.linspace(0, total_days, total_days + 1)
    S = np.zeros(total_days + 1)
    I = np.zeros(total_days + 1)
    R = np.zeros(total_days + 1)
    
    # Set initial conditions
    S[0] = population - initial_infected
    I[0] = initial_infected
    R[0] = 0
    
    # Define differential equations
    def dSdt(t, S, I):
        return -beta * S * I / population
    
    def dIdt(t, S, I):
        return beta * S * I / population - gamma * I
    
    def dRdt(t, I):
        return gamma * I
    
    # Apply RK3 method
    for i in range(total_days):
        dt = t[i+1] - t[i]
        k1_S = dSdt(t[i], S[i], I[i])
        k1_I = dIdt(t[i], S[i], I[i])
        k1_R = dRdt(t[i], I[i])
        
        k2_S = dSdt(t[i] + 0.5 * dt, S[i] + 0.5 * dt * k1_S, I[i] + 0.5 * dt * k1_I)
        k2_I = dIdt(t[i] + 0.5 * dt, S[i] + 0.5 * dt * k1_S, I[i] + 0.5 * dt * k1_I)
        k2_R = dRdt(t[i] + 0.5 * dt, I[i] + 0.5 * dt * k1_I)
        
        k3_S = dSdt(t[i] + dt, S[i] - dt * k1_S + 2 * dt * k2_S, I[i] - dt * k1_I + 2 * dt * k2_I)
        k3_I = dIdt(t[i] + dt, S[i] - dt * k1_S + 2 * dt * k2_S, I[i] - dt * k1_I + 2 * dt * k2_I)
        k3_R = dRdt(t[i] + dt, I[i] - dt * k1_I + 2 * dt * k2_I)
        
        S[i+1] = S[i] + dt / 6 * (k1_S + 4 * k2_S + k3_S)
        I[i+1] = I[i] + dt / 6 * (k1_I + 4 * k2_I + k3_I)
        R[i+1] = R[i] + dt / 6 * (k1_R + 4 * k2_R + k3_R)
        
    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}
