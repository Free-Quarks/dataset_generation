import numpy as np
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return([dS_dt, dI_dt, dR_dt])


def run_SIR_model(S0, I0, R0, beta, gamma, t_max):
    t = np.linspace(0, t_max, 10000)
    y0 = [S0, I0, R0]
    result = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = result.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered/Deceased')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

run_SIR_model(S0, I0, R0, beta, gamma, t_max)
