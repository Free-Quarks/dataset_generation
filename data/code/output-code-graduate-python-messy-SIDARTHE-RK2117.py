def sidarthe_model(beta, gamma, k, epsilon, sigma, alpha, rho, N, I0, D0, A0, R0, T): 
    t = np.linspace(0, T, T) 
    dt = T / T 
    S = N - (I0 + D0 + A0 + R0 + T) 
    I = I0 
    D = D0 
    A = A0 
    R = R0 
    T = T 
    for i in range(T-1): 
        S_next = S[i] - (beta * S[i] * I[i]/N + epsilon * S[i] * A[i]/N + rho * S[i] * T[i]/N) * dt 
        I_next = I[i] + ((beta * S[i] * I[i]/N + epsilon * S[i] * A[i]/N + rho * S[i] * T[i]/N) - (gamma + alpha + sigma)) * dt 
        D_next = D[i] + (alpha * I[i] - k * D[i]) * dt 
        A_next = A[i] + (sigma * I[i] - k * A[i]) * dt 
        R_next = R[i] + (gamma * I[i] + k * (D[i] + A[i])) * dt 
        T_next = T[i] + (k * (D[i] + A[i])) * dt 
        S.append(S_next) 
        I.append(I_next) 
        D.append(D_next) 
        A.append(A_next) 
        R.append(R_next) 
        T.append(T_next) 
    return S, I, D, A, R, T
