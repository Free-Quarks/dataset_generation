def sir_model_RK2(beta, gamma, N, I0, R0, T):
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize the arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)

    # Set the initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Define the step size
    dt = t[1] - t[0]

    # Runge-Kutta 2 method
    for i in range(T):
        k1_S = -beta * S[i] * I[i] / N
        k1_I = beta * S[i] * I[i] / N - gamma * I[i]
        k2_S = -beta * (S[i] + dt/2 * k1_S) * (I[i] + dt/2 * k1_I) / N
        k2_I = beta * (S[i] + dt/2 * k1_S) * (I[i] + dt/2 * k1_I) / N - gamma * (I[i] + dt/2 * k1_I)
        S[i+1] = S[i] + dt * (k1_S + k2_S) / 2
        I[i+1] = I[i] + dt * (k1_I + k2_I) / 2
        R[i+1] = R[i] + dt * gamma * (I[i] + dt/2 * k1_I)

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
