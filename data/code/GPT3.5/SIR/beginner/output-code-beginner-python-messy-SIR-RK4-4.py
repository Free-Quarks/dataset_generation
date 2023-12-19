def SIR_RK4(beta: float, gamma: float, population: float, infected_0: float, recovered_0: float, timesteps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    def SIR_deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.linspace(0, timesteps, timesteps + 1)
    y0 = (population - infected_0, infected_0, recovered_0)
    solution = odeint(SIR_deriv, y0, t, args=(beta, gamma))
    S, I, R = solution.T
    return t, S, I, R
