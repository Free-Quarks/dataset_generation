import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, I0, R0, H0, T0, E0, d, theta, sigma, gamma1, gamma2, alpha, beta, delta, mu, tf):
    def f(t, y):
        S, I, D, A, R, T, H, E = y
        return [-beta*S*(I + theta*A)/N,
                beta*S*(I + theta*A)/N - (sigma + alpha)*E,
                alpha*E - (gamma1 + mu)*I,
                sigma*E - (gamma2 + delta)*A,
                gamma1*I,
                gamma2*A,
                delta*A,
                d*H + mu*I]
    
    y0 = [N - I0 - R0 - H0 - T0 - E0, I0, 0, 0, R0, T0, H0, E0]
    
    t = np.linspace(0, tf, tf + 1)
    sol = np.zeros((tf + 1, 8))
    sol[0] = y0
    
    for i in range(tf):
        k1 = f(t[i], sol[i])
        k2 = f(t[i] + 1, sol[i] + k1)
        sol[i + 1] = sol[i] + (k1 + k2)/2
    
    return sol[:, 1:]

