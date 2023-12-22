def simulate_sir_rk2(S0,I0,R0,beta,gamma,dt,steps):
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0
    N = S0+I0+R0
    for step in range(1, steps):
        F0_S = -beta*I[step-1]*S[step-1]/N
        F0_I = beta*I[step-1]*S[step-1]/N - gamma*I[step-1]
        F0_R = gamma*I[step-1]
        S_half_step = S[step-1] + F0_S*dt/2
        I_half_step = I[step-1] + F0_I*dt/2
        R_half_step = R[step-1] + F0_R*dt/2
        F1_S = -beta*I_half_step*S_half_step/N
        F1_I = beta*I_half_step*S_half_step/N - gamma*I_half_step
        F1_R = gamma*I_half_step
        S[step] = S[step-1] + F1_S*dt
        I[step] = I[step-1] + F1_I*dt
        R[step] = R[step-1] + F1_R*dt
    plt.plot(S, label="Susceptible")
    plt.plot(I, label="Infected")
    plt.plot(R, label="Recovered")
    plt.legend()
    plt.show()
simulate_sir_rk2(999,1,0,0.3,0.1,0.1,1000)
