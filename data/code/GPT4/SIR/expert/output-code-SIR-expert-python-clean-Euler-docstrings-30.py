{
    "code": 
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import json

    def sir_model_euler(S0, I0, R0, beta, gamma, num_days):
        ''''
        This function simulates the SIR model using the Euler method.

        Parameters:
        S0 (float): Initial number of susceptibles.
        I0 (float): Initial number of infected.
        R0 (float): Initial number of recovered.
        beta (float): The parameter controlling how often a susceptible-infected contact results in a new infection.
        gamma (float): The rate an infected recovers and moves into the resistant phase.
        num_days (int): The number of days.

        Returns:
        dict: A dictionary with susceptible, infected and recovered individuals.
        ''''
        
        # Initialize the population
        S, I, R = [S0], [I0], [R0]
        N = S0 + I0 + R0

        dt = 1.0
        
        for _ in range(num_days):
            next_S = S[-1] - (beta*S[-1]*I[-1]/N)*dt
            next_I = I[-1] + (beta*S[-1]*I[-1]/N - gamma*I[-1])*dt
            next_R = R[-1] + (gamma*I[-1])*dt

            S.append(next_S)
            I.append(next_I)
            R.append(next_R)

        # Plotting the data
        days = range(num_days + 1)
        plt.figure(figsize=(6,4))
        plt.plot(days, S, label='Susceptible')
        plt.plot(days, I, label='Infected')
        plt.plot(days, R, label='Recovered')
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Number of individuals')
        plt.grid(True)
        plt.show()

        return {'Susceptible': S, 'Infected': I, 'Recovered': R}

    # Usage 
    sir_model_euler(999, 1, 0, 0.3, 0.1, 160)
    "",
    "function_name": "sir_model_euler"
}
