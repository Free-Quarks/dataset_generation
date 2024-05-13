import numpy as np
import matplotlib.pyplot as plt

def seir_rk2(N, beta, gamma, sigma, initial_infected, initial_exposed, initial_recovered, t_max, dt):
    # Total population
    S0 = N - initial_infected - initial_exposed - initial_recovered
    # Initial conditions
    S = [S0]
    E = [initial_exposed]
    I = [initial_infected]
    R = [initial_recovered]
    # Time vector
    t = np.arange(0, t_max, dt)
    # Runge-Kutta method
    for i in range(len(t)-1):
        dSdt = -beta/N * S[i] * I[i]
        dEdt = beta/N * S[i] * I[i] - sigma * E[i]
        dIdt = sigma * E[i] - gamma * I[i]
        dRdt = gamma * I[i]
        S_next = S[i] + dt * (dSdt + (-beta/N * (S[i] + dt/2 * dSdt) * (I[i] + dt/2 * dIdt))) / 2
        E_next = E[i] + dt * (dEdt + (beta/N * (S[i] + dt/2 * dSdt) * (I[i] + dt/2 * dIdt)) - sigma * (E[i] + dt/2 * dEdt)) / 2
        I_next = I[i] + dt * (dIdt + (sigma * (E[i] + dt/2 * dEdt) - gamma * (I[i] + dt/2 * dIdt))) / 2
        R_next = R[i] + dt * (dRdt + gamma * (I[i] + dt/2 * dIdt)) / 2
        S.append(S_next)
        E.append(E_next)
        I.append(I_next)
        R.append(R_next)
    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
}
