def sir_rk2(beta, gamma, population, i0, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    N = len(t)
    s = np.zeros(N)
    i = np.zeros(N)
    r = np.zeros(N)
    
    # Set initial conditions
    s[0] = population - i0
    i[0] = i0
    r[0] = 0
    
    # Runge-Kutta 2nd Order
    for n in range(N-1):
        k1_s = -beta * s[n] * i[n] / population
        k1_i = beta * s[n] * i[n] / population - gamma * i[n]
        k2_s = -beta * (s[n] + 0.5 * dt * k1_s) * (i[n] + 0.5 * dt * k1_i) / population
        k2_i = beta * (s[n] + 0.5 * dt * k1_s) * (i[n] + 0.5 * dt * k1_i) / population - gamma * (i[n] + 0.5 * dt * k1_i)
        
        s[n+1] = s[n] + dt * k2_s
        i[n+1] = i[n] + dt * k2_i
        r[n+1] = population - s[n+1] - i[n+1]
    
    # Return results
    return s, i, r
