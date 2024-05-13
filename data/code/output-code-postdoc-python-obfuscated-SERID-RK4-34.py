import numpy as np
import matplotlib.pyplot as plt

def simulate_serid(beta, sigma, gamma, N, I0, R0, D0, t_max):
    def deriv(y, t, N, beta, sigma, gamma):
        S, E, RI, RD, D = y
        N = S + E + RI + RD + D
        dSdt = -beta * S * (RI + RD) / N
        dEdt = beta * S * (RI + RD) / N - sigma * E
        dRIdt = sigma * E - gamma * RI
        dRDdt = sigma * E - gamma * RD
        dDdt = gamma * (RI + RD)
        return dSdt, dEdt, dRIdt, dRDdt, dDdt

    t = np.linspace(0, t_max, t_max + 1)
    y0 = N - I0 - R0 - D0, 0, I0, R0, D0
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
    S, E, RI, RD, D = ret.T

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, RI, 'g', label='Recovered (Infected)')
    ax.plot(t, RD, 'm', label='Recovered (Dead)')
    ax.plot(t, D, 'r', label='Deceased')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title('SERID Model Simulation')
    ax.legend()
    plt.show()


# Example usage
def main():
    beta = 0.5
    sigma = 1 / 3
    gamma = 1 / 14
    N = 1000
    I0 = 1
    R0 = 0
    D0 = 0
    t_max = 200
    simulate_serid(beta, sigma, gamma, N, I0, R0, D0, t_max)


if __name__ == '__main__':
    main()
