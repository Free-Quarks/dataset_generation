import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def SIR_Model(y, t, N, beta, gamma):
    """
    SIR model function.
    Args:
        y (tuple): tuple with three values (susceptible, infected, removed).
        t (float): time.
        N (int): total population.
        beta (float): transmission rate parameter.
        gamma (float): recovery rate parameter.
    Returns:
        dsdt: change of susceptible individuals with respect to time.
        didt: change of infected individuals with respect to time.
        drdt: change of removed individuals with respect to time.
    """
    S, I, R = y
    dsdt = -beta * S * I / N
    didt = beta * S * I / N - gamma * I
    drdt = gamma * I
    return dsdt, didt, drdt
