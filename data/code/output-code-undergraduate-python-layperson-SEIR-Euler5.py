import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, E0, R0, t_end):
    
    # Initial conditions
    S0 = N - I0 - E0 - R0
    
    # Step size
    dt = 0.01
    
    # Time array
    t = np.arange(0, t_end, dt)
    
    # Initialize arrays
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Euler's method to solve the differential equations
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dS
        E[i] = E[i-1] + dt * dE
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    
    # Plotting
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
}

