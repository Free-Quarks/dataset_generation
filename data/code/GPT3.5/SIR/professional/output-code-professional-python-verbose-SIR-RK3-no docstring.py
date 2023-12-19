import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, population, initial_infected, days):
    # Define the differential equations
    def sir_equations(y, t):
        S, I, R = y
        dSdt = -beta * S * I / population
        dIdt = beta * S * I / population - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Set initial conditions
    initial_conditions = [population - initial_infected, initial_infected, 0]

    # Set time steps
    t = np.linspace(0, days, int(days / 0.1))

    # Solve the differential equations using RK3
    sol = ode.RK3(sir_equations, initial_conditions, t)

    # Extract the solution
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]

    # Plot the solution
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
population = 1000
initial_infected = 1
days = 100

sir_model(beta, gamma, population, initial_infected, days)
