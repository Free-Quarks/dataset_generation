def sir_model(beta, gamma, s0, i0, r0, t_max, dt):
    # Initialize lists to store the values
    t = [0]
    S = [s0]
    I = [i0]
    R = [r0]
    # Euler's method to solve the differential equations
    for _ in range(int(t_max/dt)):
        t.append(t[-1] + dt)
        s_next = S[-1] - (beta*S[-1]*I[-1])*dt
        i_next = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        r_next = R[-1] + (gamma*I[-1])*dt
        S.append(s_next)
        I.append(i_next)
        R.append(r_next)
    # Return the lists of values
    return t, S, I, R
