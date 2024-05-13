import numpy as np
import matplotlib.pyplot as plt


def seird_model(beta=0.3, sigma=0.1, gamma=0.1, mu=0.01, N=1000, I0=1, E0=0, R0=0, D0=0, T=365):
    def derivs(y, t):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - (gamma + mu) * I
        dRdt = gamma * I
        dDdt = mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt
    
    t = np.linspace(0, T, T+1)
    y0 = N - I0 - E0 - R0 - D0
    y0 = y0, E0, I0, R0, D0
    ode_result = odeint(derivs, y0, t)
    S, E, I, R, D = ode_result.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIRD Model Simulation')
    plt.legend()
    plt.show()


seird_model(beta=0.3, sigma=0.1, gamma=0.1, mu=0.01, N=1000, I0=1, E0=0, R0=0, D0=0, T=365)
