def serid_rk3(beta, gamma, mu, N, I0, R0, D0, t_end, dt):
    
    def dSdt(t, N, S, I, R, D):
        return -beta * S * (I / N)
    
    def dIdt(t, N, S, I, R, D):
        return (beta * S * (I / N)) - (gamma * I) - (mu * I)
    
    def dRdt(t, N, S, I, R, D):
        return gamma * I
    
    def dDdt(t, N, S, I, R, D):
        return mu * I
    
    def serid_model(t, y):
        S, I, R, D = y
        dS = dSdt(t, N, S, I, R, D)
        dI = dIdt(t, N, S, I, R, D)
        dR = dRdt(t, N, S, I, R, D)
        dD = dDdt(t, N, S, I, R, D)
        return [dS, dI, dR, dD]
    
    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    y0 = [N-I0-R0-D0, I0, R0, D0]
    sol = solve_ivp(serid_model, [0, t_end], y0, t_eval=t)
    S, I, R, D = sol.y
    
    return t, S, I, R, D
