import numpy as np
import matplotlib.pyplot as plt
import json

def rk4(t, dt, n, beta, gamma, S, I, R):
    f1 = dt * (-beta * S * I / n)
    g1 = dt * (beta * S * I / n - gamma * I)
    h1 = dt * gamma * I

    f2 = dt * (-beta * (S + 0.5 * f1) * (I + 0.5 * g1) / n)
    g2 = dt * (beta * (S + 0.5 * f1) * (I + 0.5 * g1) / n - gamma * (I + 0.5 * g1))
    h2 = dt * gamma * (I + 0.5 * g1)

    f3 = dt * (-beta * (S + 0.5 * f2) * (I + 0.5 * g2) / n)
    g3 = dt * (beta * (S + 0.5 * f2) * (I + 0.5 * g2) / n - gamma * (I + 0.5 * g2))
    h3 = dt * gamma * (I + 0.5 * g2)

    f4 = dt * (-beta * (S + f3) * (I + g3) / n)
    g4 = dt * (beta * (S + f3) * (I + g3) / n - gamma * (I + g3))
    h4 = dt * gamma * (I + g3)

    S += (f1 + 2 * f2 + 2 * f3 + f4) / 6
    I += (g1 + 2 * g2 + 2 * g3 + g4) / 6
    R += (h1 + 2 * h2 + 2 * h3 + h4) / 6

    return S, I, R
