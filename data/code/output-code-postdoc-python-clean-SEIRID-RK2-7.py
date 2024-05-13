import numpy as np
import matplotlib.pyplot as plt

def seirid_model(N, beta, gamma, delta, alpha, rho, t_max, dt):
    S = [N-1]
    E = [0]
    I = [1]
    R = [0]
    D = [0]
    t = np.arange(0, t_max, dt)
    for i in range(len(t)-1):
        dSdt = -beta*S[i]*I[i]/N
        dEdt = beta*S[i]*I[i]/N - delta*E[i] - alpha*E[i]
        dIdt = delta*E[i] - gamma*I[i] - rho*I[i]
        dRdt = gamma*I[i]
        dDdt = alpha*E[i] + rho*I[i]
        S.append(S[i] + dt*dSdt)
        E.append(E[i] + dt*dEdt)
        I.append(I[i] + dt*dIdt)
        R.append(R[i] + dt*dRdt)
        D.append(D[i] + dt*dDdt)
    return S, E, I, R, D

