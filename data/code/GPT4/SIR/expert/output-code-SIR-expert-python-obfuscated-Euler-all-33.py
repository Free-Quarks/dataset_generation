import numpy as np
import matplotlib.pyplot as plt
import json
def bPp(b, g, N, I0, R0, S0, t):
    t_= np.linspace(0, t, t)
    S, I, R = [S0], [I0], [R0]
    fS, fI, fR = lambda s,i: -b*s*i/N, lambda s,i: b*s*i/N-g*i, lambda i: g*i
    for _ in t_[1:]:
        nS = S[-1] + fS(S[-1], I[-1])
        nI = I[-1] + fI(S[-1], I[-1])
        nR = R[-1] + fR(I[-1])
        S.append(nS)
        I.append(nI)
        R.append(nR)
    plt.plot(t_, S, 'b', label='Susceptible')
    plt.plot(t_, I, 'r', label='Infected')
    plt.plot(t_, R, 'g', label='Recovered')
    plt.legend()
    plt.show()
bPp(0.2, 0.1, 1000, 1, 0, 999, 160)
