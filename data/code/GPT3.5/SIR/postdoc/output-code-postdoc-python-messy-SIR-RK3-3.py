import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, R0, T):
    def SIR_model(y, t): # SIR model
        S, I, R = y
        dS = -beta*S*I/N
        dI = beta*S*I/N - gamma*I
        dR = gamma*I
        return [dS, dI, dR]

    t = np.linspace(0, T, 1000)
    y0 = [N-I0-R0, I0, R0] # initial conditions

    # Solve the ODE system
    sol = odeint(SIR_model, y0, t)
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}
