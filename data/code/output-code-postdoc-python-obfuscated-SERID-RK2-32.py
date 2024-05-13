import numpy as np
import matplotlib.pyplot as plt

def serid_rk2(S0, E0, R0, I0, D0, beta, gamma, delta, alpha, N, dt, T):
    def derivs(state, t):
        S, E, R, I, D = state
        dSdt = -beta*S*I/N
        dEdt = beta*S*I/N - delta*E
        dRdt = gamma*I
        dIdt = delta*E - (gamma + alpha)*I
        dDdt = alpha*I
        return [dSdt, dEdt, dRdt, dIdt, dDdt]

    state0 = [S0, E0, R0, I0, D0]
    t = np.linspace(0, T, int(T/dt))
    states = odeint(derivs, state0, t)
    S, E, R, I, D = states.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

serid_rk2(500, 1, 0, 1, 0, 0.3, 0.1, 0.1, 0.01, 1000, 0.1, 100)
