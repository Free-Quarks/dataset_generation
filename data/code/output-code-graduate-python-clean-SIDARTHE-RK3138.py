import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, sigma, gamma, mu, delta, alpha, rho, population, initial_infected, days):
    # Parameters
    N = population
    S = N - initial_infected
    I = initial_infected
    D = 0
    A = 0
    R = 0
    T = 0
    H = 0
    E = 0
    t = 0
    dt = 1
    timeline = [t]
    susceptible = [S]
    infected = [I]
    deceased = [D]
    asymptomatic = [A]
    recovered = [R]
    tested = [T]
    hospitalized = [H]
    exposed = [E]
    
    # Run simulation
    while t < days:
        ds = -beta * S * (I + alpha * A) / N
        de = beta * S * (I + alpha * A) / N - sigma * E
        di = sigma * (1 - mu) * E - (rho + gamma) * I
        dd = sigma * mu * E
        da = alpha * sigma * (1 - rho) * E - delta * A
        dr = gamma * I
        dt = delta * A
        dh = sigma * rho * E
        
        S += ds * dt
        E += de * dt
        I += di * dt
        D += dd * dt
        A += da * dt
        R += dr * dt
        T += dt * dt
        H += dh * dt
        
        t += dt
        
        timeline.append(t)
        susceptible.append(S)
        exposed.append(E)
        infected.append(I)
        deceased.append(D)
        asymptomatic.append(A)
        recovered.append(R)
        tested.append(T)
        hospitalized.append(H)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(timeline, susceptible, label='Susceptible')
    plt.plot(timeline, exposed, label='Exposed')
    plt.plot(timeline, infected, label='Infected')
    plt.plot(timeline, deceased, label='Deceased')
    plt.plot(timeline, asymptomatic, label='Asymptomatic')
    plt.plot(timeline, recovered, label='Recovered')
    plt.plot(timeline, tested, label='Tested')
    plt.plot(timeline, hospitalized, label='Hospitalized')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIDARTHE Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()
}

