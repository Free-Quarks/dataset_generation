import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, T):
    # Total population
    S0 = N - I0 - R0
    
    # Initial conditions vector
    y0 = S0, I0, R0
    
    # Time vector
    t = np.linspace(0, T, T+1)
    
    # Differential equations
    def SIR_eqn(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Runge-Kutta method (second order)
    def RK2_step(y, t, h, beta, gamma):
        k1 = h * SIR_eqn(y, t, beta, gamma)
        k2 = h * SIR_eqn(y + 0.5 * k1, t + 0.5 * h, beta, gamma)
        y_next = y + k2
        return y_next
    
    # Solve the differential equations
    S, I, R = [S0], [I0], [R0]
    h = t[1] - t[0]
    for i in range(T):
        y_next = RK2_step((S[i], I[i], R[i]), t[i], h, beta, gamma)
        S.append(y_next[0])
        I.append(y_next[1])
        R.append(y_next[2])
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()
}

