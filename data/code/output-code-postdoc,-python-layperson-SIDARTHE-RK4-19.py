import numpy as np
import matplotlib.pyplot as plt

# Define the SIDARTHE model

def sidarthe_model(N, E0, I0, D0, A0, R0, T0, beta, sigma, gamma, theta, delta, kappa, epsilon, rho, alpha):
    # Total population
    S0 = N - (E0 + I0 + D0 + A0 + R0 + T0)
    
    # Initial conditions
    Y0 = S0, E0, I0, D0, A0, R0, T0
    
    def deriv(Y, t, N, beta, sigma, gamma, theta, delta, kappa, epsilon, rho, alpha):
        S, E, I, D, A, R, T = Y
        dSdt = -beta * S * (I + theta * A) / N
        dEdt = beta * S * (I + theta * A) / N - sigma * E
        dIdt = sigma * E - (gamma + delta) * I
        dDdt = delta * I - (kappa + epsilon) * D
        dAdt = epsilon * D - (rho + alpha) * A
        dRdt = gamma * I + rho * A
        dTdt = kappa * D + alpha * A
        return dSdt, dEdt, dIdt, dDdt, dAdt, dRdt, dTdt
    
    # Time vector
    t = np.linspace(0, 365, num=365)
    
    # Integrate the SIDARTHE equations over the time grid
    ret = odeint(deriv, Y0, t, args=(N, beta, sigma, gamma, theta, delta, kappa, epsilon, rho, alpha))
    S, E, I, D, A, R, T = ret.T
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, D, 'g', label='Deceased')
    ax.plot(t, A, 'm', label='Affected')
    ax.plot(t, R, 'c', label='Recovered')
    ax.plot(t, T, 'k', label='Treated')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('SIDARTHE Model')
    ax.legend()
    plt.show()
    
# Call the SIDARTHE model function
sidarthe_model(N, E0, I0, D0, A0, R0, T0, beta, sigma, gamma, theta, delta, kappa, epsilon, rho, alpha)
