import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(N, I0, R0, beta, gamma, num_days):
    # Initial conditions
    S0 = N - I0 - R0
    Y0 = np.array([S0, I0, R0])
    
    # Time vector
    t = np.linspace(0, num_days, num_days)
    
    # Function to solve the system of ODEs
    def sir_model(Y, t, N, beta, gamma):
        S, I, R = Y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    # Solve the system of ODEs
    sol = odeint(sir_model, Y0, t, args=(N, beta, gamma))
    
    # Plot the results
    plt.plot(t, sol[:, 0], 'b', label='Susceptible')
    plt.plot(t, sol[:, 1], 'r', label='Infected')
    plt.plot(t, sol[:, 2], 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.grid(True)
    plt.show()
}
