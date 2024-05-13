import numpy as np

def serid_model(init_vals, params, t):
    S, E, I, R, D = init_vals
    S_out = np.zeros(t)
    E_out = np.zeros(t)
    I_out = np.zeros(t)
    R_out = np.zeros(t)
    D_out = np.zeros(t)
    S_out[0] = S
    E_out[0] = E
    I_out[0] = I
    R_out[0] = R
    D_out[0] = D

    def derivs(y, t, N, beta, sigma, gamma, mu):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - (1 - mu) * gamma * I - mu * D
        dRdt = (1 - mu) * gamma * I
        dDdt = mu * D
        return dSdt, dEdt, dIdt, dRdt, dDdt

    for i in range(1, t):
        y = S, E, I, R, D
        S_out[i], E_out[i], I_out[i], R_out[i], D_out[i] = rk4_step(y, t[i-1], params, derivs)
    return S_out, E_out, I_out, R_out, D_out


def rk4_step(y, t, params, derivs):
    h = t[1] - t[0]
    k1 = h * derivs(y, t, *params)
    k2 = h * derivs(y + 0.5 * k1, t + 0.5 * h, *params)
    k3 = h * derivs(y + 0.5 * k2, t + 0.5 * h, *params)
    k4 = h * derivs(y + k3, t + h, *params)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def main():
    init_vals = 1000, 1, 0, 0, 0
    params = 1000, 0.2, 1/14, 0.02
    t = np.linspace(0, 100, 100)
    S, E, I, R, D = serid_model(init_vals, params, t)
    
    # Plotting the results
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.plot(t, D, 'k', label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SERID Model Simulation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
