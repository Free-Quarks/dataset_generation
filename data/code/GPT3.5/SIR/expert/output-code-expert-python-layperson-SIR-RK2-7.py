import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, T, N):
    def SIR_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    t = np.linspace(0, T, 1000)
    y0 = [S0, I0, R0]
    
    sol = odeint(SIR_model, y0, t)
    S, I, R = sol.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.grid()
    plt.show()


# Example usage:
# SIR_RK2(0.3, 0.1, 900, 100, 0, 100, 1000)
