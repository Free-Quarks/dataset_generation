import numpy as np
import matplotlib.pyplot as plt


def seird_model(N, I0, E0, R0, D0, beta, sigma, gamma, delta, days):
    # total population
    S0 = N - I0 - E0 - R0 - D0
    
    # arrays to store the compartment sizes
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    D = np.zeros(days)
    
    # set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    
    # euler method to solve the differential equations
    for d in range(days-1):
        S[d+1] = S[d] - beta*S[d]*I[d]/N
        E[d+1] = E[d] + beta*S[d]*I[d]/N - sigma*E[d]
        I[d+1] = I[d] + sigma*E[d] - gamma*I[d] - delta*I[d]
        R[d+1] = R[d] + gamma*I[d]
        D[d+1] = D[d] + delta*I[d]
    
    # plot the results
    t = np.arange(days)
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.plot(t, D, 'm', label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIRD Model')
    plt.legend()
    plt.show()


seird_model(10000, 1, 0, 0, 0, 0.5, 0.2, 0.1, 0.05, 100)
