import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, T):
    def SIR_model(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    y0 = N - I0, I0, 0
    t = np.linspace(0, T, T+1)
    
    sol = odeint(SIR_model, y0, t)
    
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
}
