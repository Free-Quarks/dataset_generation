import numpy as p, matplotlib.pyplot as pl
def SIR_Euler(beta, gamma, S0, I0, R0, T, dt):
    """
    Simulate the SIR model using Euler's method
    Parameters:
    beta: The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma: The rate an infected recovers and moves into the resistant phase.
    S0: The number of susceptible individuals at the start of the outbreak.
    I0: The number of infected individuals at the start of the outbreak.
    R0: The number of recovered individuals at the start of the outbreak.
    T: Total time to simulate.
    dt: The time step.
    """
    N = S0 + I0 + R0
    t = p.arange(0, T, dt)
    S = p.zeros(len(t)); I = p.zeros(len(t)); R = p.zeros(len(t))
    S[0] = S0; I[0] = I0; R[0] = R0
    for i in range(len(t)-1):
        S[i+1] = S[i] - (beta*S[i]*I[i]/N)*dt
        I[i+1] = I[i] + (beta*S[i]*I[i]/N - gamma*I[i])*dt
        R[i+1] = R[i] + (gamma*I[i])*dt
    pl.figure(figsize=(6,4)); pl.plot(t, S, label='S(t)'); pl.plot(t, I, label='I(t)');  pl.plot(t, R, label='R(t)'); pl.legend(); pl.show()

SIR_Euler(0.1, 0.05, 1000, 1, 0, 200, 0.1)
