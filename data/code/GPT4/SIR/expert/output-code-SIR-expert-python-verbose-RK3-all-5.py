# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function for the SIR model differential equations.
def deriv_SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt
