import numpy as np
import matplotlib.pyplot as plt
def sir_model(t, y, beta, gamma):
    S, I, R = y
    N = S + I + R
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

def runge_kutta_2(f, t, y, h, beta, gamma):
    k1 = f(t, y, beta, gamma)
    k2 = f(t + h, y + h * k1, beta, gamma)
    y_next = y + h * (k1 + k2) / 2.0
    return y_next
