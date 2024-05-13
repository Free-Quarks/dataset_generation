import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, E0, I0, R0, T_inc, T_inf, T_hosp, T_crit, T_rec, T_death, p_severe, p_critical, p_fatal, beta, gamma, delta, delta_h, delta_c):
    def derivs(state, t):
        S, E, I, R, D, H, C = state
        dSdt = -beta * S * (I + p_severe * H + p_critical * C) / N
        dEdt = beta * S * (I + p_severe * H + p_critical * C) / N - delta * E
        dIdt = delta * E - (1 - p_severe) * gamma * I - p_severe * delta_h * H - p_severe * delta_c * C
        dRdt = (1 - p_severe) * gamma * I - (1 - p_critical) * T_rec * R
        dDdt = p_fatal * delta_c * C
        dHdt = p_severe * delta_h * H - (1 - p_fatal) * T_hosp * H
        dCdt = p_severe * delta_c * C - T_crit * C
        return [dSdt, dEdt, dIdt, dRdt, dDdt, dHdt, dCdt]

    state0 = [N - E0 - I0 - R0, E0, I0, R0, 0, 0, 0]
    t = np.linspace(0, 100, 100)
    y = odeint(derivs, state0, t)

    S = y[:, 0]
    E = y[:, 1]
    I = y[:, 2]
    R = y[:, 3]
    D = y[:, 4]
    H = y[:, 5]
    C = y[:, 6]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deaths')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, C, label='Critical')
    plt.legend()
    plt.show()
}
