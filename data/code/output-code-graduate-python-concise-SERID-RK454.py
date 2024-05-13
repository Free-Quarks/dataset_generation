def serid_model(beta, gamma, N, I0, R0, D0, timesteps):
    # Define the differential equations
    def deriv(y, t, beta, gamma, N):
        S, E, R, I, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - gamma * E
        dRdt = gamma * E
        dIdt = gamma * E - D * I
        dDdt = D * I
        return dSdt, dEdt, dRdt, dIdt, dDdt

    # Set initial conditions
    S0 = N - I0 - R0 - D0
    E0 = 0

    # Set time grid
    t = np.linspace(0, timesteps, timesteps)

    # Integrate the SEIRD equations over the time grid
    y0 = S0, E0, R0, I0, D0
    ret = odeint(deriv, y0, t, args=(beta, gamma, N))
    S, E, R, I, D = ret.T

    # Return the results
    return S, E, R, I, D
