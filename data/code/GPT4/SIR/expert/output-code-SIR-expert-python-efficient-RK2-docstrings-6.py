import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model_RK2(S0, I0, R0, beta, gamma, T, dt):
    N = S0 + I0 + R0
    t = np.arange(0, T, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0], I[0], R[0] = S0, I0, R0

    def dSdt(S, I):
        return -beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    for i in range(len(t) - 1):
        k1_S = dt * dSdt(S[i], I[i])
        k1_I = dt * dIdt(S[i], I[i])
        k1_R = dt * dRdt(I[i])

        k2_S = dt * dSdt(S[i] + 0.5 * k1_S, I[i] + 0.5 * k1_I)
        k2_I = dt * dIdt(S[i] + 0.5 * k1_S, I[i] + 0.5 * k1_I)
        k2_R = dt * dRdt(I[i] + 0.5 * k1_I)

        S[i + 1] = S[i] + k2_S
        I[i + 1] = I[i] + k2_I
        R[i + 1] = R[i] + k2_R

    return S, I, R, t
