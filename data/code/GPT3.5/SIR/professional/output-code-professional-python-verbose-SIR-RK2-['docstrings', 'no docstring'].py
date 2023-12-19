import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    """
    Simulate and plot the SIR model using the Runge-Kutta 2nd order method.
    
    Parameters:
    - beta: float, the infection rate
    - gamma: float, the recovery rate
    - S0: float, the initial susceptible population
    - I0: float, the initial infected population
    - R0: float, the initial recovered population
    - t_max: float, the maximum time to simulate
    - dt: float, the time step size
    """
    
    def f(t, S, I, R):
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.arange(0, t_max, dt)
    S, I, R = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    
    S[0], I[0], R[0] = S0, I0, R0
    
    for i in range(1, len(t)):
        h = dt
        h2 = h / 2
        
        k1 = f(t[i - 1], S[i - 1], I[i - 1], R[i - 1])
        k2 = f(t[i - 1] + h2, S[i - 1] + h2 * k1[0], I[i - 1] + h2 * k1[1], R[i - 1] + h2 * k1[2])
        S[i] = S[i - 1] + h * k2[0]
        I[i] = I[i - 1] + h * k2[1]
        R[i] = R[i - 1] + h * k2[2]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.grid(True)
    plt.show()
}
