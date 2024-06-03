from itertools import product

import numpy as np


def diff_sinc(w1, w2):
    if np.isclose(w1, w2):
        return 1.0
    else:
        return np.sin(w1 - w2) / (w1 - w2)


def overlap(w: np.ndarray):
    n = w.shape[0]
    s_mat = np.zeros((n, n))
    for i, j in product(range(n), repeat=2):
        s_mat[i, j] = diff_sinc(w[i], w[j])
    return s_mat


def uniform_w(n):
    return np.arange(n) * np.pi


def quadratic_w(n):
    mx = n * np.pi
    return mx * (np.arange(n) / n) ** 2


def random_w(n):
    mx = n * np.pi
    return np.sort(np.random.uniform(mx, size=n))


def matrix_analysis(m: np.ndarray):
    s, _ = np.linalg.eig(m)
    s = np.abs(s)
    return np.min(s), np.max(s)


if __name__ == "__main__":
    n = 10
    trial = 10000
    u_w, q_w = uniform_w(n), quadratic_w(n)
    print(f"n={n}")
    print("\tUniform", end='\n\t')
    print(matrix_analysis(overlap(u_w)))
    print("\tQuadratic", end='\n\t')
    print(matrix_analysis(overlap(q_w)))
    print("\tRandom", end='\n\t')
    s_min_list = list()
    for _ in range(10000):
        r_w = random_w(n)
        s_min, s_max = matrix_analysis(overlap(r_w))
        s_min_list.append(s_min)
    print(f"mean = {np.mean(s_min_list)}, sigma = {np.std(s_min_list)}")

    """
    Output Example:
    n=10
        Uniform
        (0.9999999999999998, 1.0000000000000004)
        Quadratic
        (3.1409066991943e-05, 2.9429152729706023)
        Random
        mean = 0.0028498717998976415, sigma = 0.013115523403378994
    
    Process finished with exit code 0
    """