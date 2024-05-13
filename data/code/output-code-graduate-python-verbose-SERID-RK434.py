def serid(model_params, initial_conditions, timesteps):
    # Unpack the model parameters
    beta = model_params['beta']
    gamma = model_params['gamma']
    rho = model_params['rho']
    # Unpack the initial conditions
    S_0, E_0, I_0, R_0 = initial_conditions
    # Initialize the lists to store the results
    S = [S_0]
    E = [E_0]
    I = [I_0]
    R = [R_0]
    # Define the differential equations
    def dSdt(beta, S, I):
        return -beta * S * I
    def dEdt(beta, gamma, S, E, I):
        return beta * S * I - gamma * E
    def dIdt(gamma, rho, E, I):
        return gamma * E - rho * I
    def dRdt(rho, I):
        return rho * I
    # Iterate over the timesteps
    for t in range(timesteps):
        # Compute the derivatives
        dS = dSdt(beta, S[-1], I[-1])
        dE = dEdt(beta, gamma, S[-1], E[-1], I[-1])
        dI = dIdt(gamma, rho, E[-1], I[-1])
        dR = dRdt(rho, I[-1])
        # Update the variables using the Runge-Kutta method
        S.append(S[-1] + dS)
        E.append(E[-1] + dE)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
    # Return the results
    return S, E, I, R
