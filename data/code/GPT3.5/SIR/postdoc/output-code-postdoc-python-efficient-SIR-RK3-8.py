import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, T):
    # Function to calculate the derivatives for SIR model
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Time vector
    t = np.linspace(0, T, T)

    # Initial conditions
    S0 = N - I0 - R0
    y0 = S0, I0, R0

    # Solve the ODE using Runge-Kutta 3rd order
    ode_sol = np.zeros((T, 3))
    ode_sol[0] = y0
    for i in range(1, T):
        k1 = deriv(ode_sol[i-1], t[i-1], N, beta, gamma)
        k2 = deriv(ode_sol[i-1] + 0.5 * k1, t[i-1] + 0.5, N, beta, gamma)
        k3 = deriv(ode_sol[i-1] - k1 + 2 * k2, t[i-1] + 1, N, beta, gamma)
        ode_sol[i] = ode_sol[i-1] + (1/6) * (k1 + 4 * k2 + k3)

    # Plotting
    plt.plot(t, ode_sol[:, 0], label='S')
    plt.plot(t, ode_sol[:, 1], label='I')
    plt.plot(t, ode_sol[:, 2], label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model using RK3')
    plt.show()
}
