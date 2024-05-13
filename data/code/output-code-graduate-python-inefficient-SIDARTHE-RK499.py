import numpy as np
import matplotlib.pyplot as plt


def sidarthe(beta, gamma, theta, delta, rho, kappa, sigma, mu, nu, N, I0, R0, D0, E0, A0, T0, H0):
    
    def deriv(y, t, N, beta, gamma, theta, delta, rho, kappa, sigma, mu, nu):
        S, I, D, A, R, T, H, E = y
        dSdt = -beta * S * (I + theta * A) / N
        dEdt = beta * S * (I + theta * A) / N - delta * E
        dIdt = delta * (1 - rho) * E - (1 - sigma) * gamma * I - sigma * mu * I
        dDdt = delta * rho * E
        dAdt = delta * (1 - rho) * E - nu * kappa * A
        dTdt = sigma * mu * I - nu * T
        dHdt = sigma * (1 - mu) * I - nu * H
        dRdt = gamma * (1 - sigma) * I + nu * (T + H)
        return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt
    
    y0 = N - I0 - R0 - D0 - E0 - A0 - T0 - H0
    t = np.arange(0, 500, 1)
    
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, theta, delta, rho, kappa, sigma, mu, nu))
    S, I, D, A, R, T, H, E = ret.T
    
    plt.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    plt.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    plt.plot(t, D/1000, 'g', alpha=0.5, lw=2, label='Deceased')
    plt.plot(t, A/1000, 'y', alpha=0.5, lw=2, label='Asymptomatic')
    plt.plot(t, R/1000, 'm', alpha=0.5, lw=2, label='Recovered')
    plt.plot(t, T/1000, 'k', alpha=0.5, lw=2, label='Treated')
    plt.plot(t, H/1000, 'c', alpha=0.5, lw=2, label='Hospitalized')
    plt.plot(t, E/1000, 'gray', alpha=0.5, lw=2, label='Exposed')
    plt.xlabel('Time (days)')
    plt.ylabel('Number (thousands)')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}

