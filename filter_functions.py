import numpy as np
from matplotlib import pyplot as plt
from scipy.special import chebyt


def omega_shifter(filter_func):
    def _omega_shifter(omega, n, delta_t, *args):
        if not n % 2 == 1:
            raise ValueError
        coeff = filter_func(n, delta_t, *args)
        coeff /= np.sum(coeff)
        n_half = n // 2
        shifter = np.exp(np.linspace(-n_half, n_half, n) * 1j * delta_t * omega)
        return coeff * shifter

    return _omega_shifter


@omega_shifter
def single_tone(n, *_):
    return np.ones(n)


@omega_shifter
def gaussian_filter(n, delta_t, delta_e, *_):
    n_half = n // 2
    delta = delta_e * delta_t
    g_coeff = delta * np.exp(-(delta * np.linspace(-n_half, n_half, n)) ** 2 / 2) / np.sqrt(2 * np.pi)
    return g_coeff


@omega_shifter
def square_filter(n, *_):
    vec = np.zeros(n, dtype=complex)
    n_half = n // 2
    for i, k in enumerate(range(-n_half, n_half + 1)):
        if abs(k) % 2 == 1:
            vec[i] = 1j / (np.pi * k)
        elif k == 0:
            vec[i] = 0.5
    return vec


@omega_shifter
def trigonometric_chebyshev_filter(n, delta_t, delta_e, *_):
    assert n % 2 == 1  # Then the corresponding chebyshev poly is odd.
    n_half = n // 2
    a = delta_t * delta_e * np.pi
    b = np.cos(a)
    if np.isclose(b, -1):
        raise ValueError

    # y = np.poly1d(np.array([-1, b]) / (1 + b))  # y = (cos a - cos x) / (cos a + 1)
    y = np.poly1d(np.array([2, 1 - b])/(1 + b))  # y = 2(cos x - cos a) / (cos a + 1) + 1
    y0 = y(1)  # y0 = y(x=0)

    # poly = -sh_chebyt(n_half)  # p = -T_n(2y-1) = T_n(1-2y)
    poly = chebyt(n_half)
    poly = np.polyval(poly, y)
    poly = poly / poly(y0)

    mon_c = poly.c[::-1]  # p(y) = Σk mon_c[k] cos^k x
    mon_by_cheby = _monomial_by_chebyshev(n_half + 1)  # cos x^k = Σj d[k, j] Tj(cos x) = Σj d[k, j] cos jx
    cheby_c = mon_c @ mon_by_cheby  # p(y) = Σk cheby_c[k] Tk(cos x) = Σk cheby_c[k] cos kx

    tot_coeff = np.zeros(n, dtype=complex)
    tot_coeff[:n_half] = cheby_c[:0:-1]/2
    tot_coeff[n_half] = cheby_c[0]
    tot_coeff[n_half + 1:] = cheby_c[1:]/2
    return tot_coeff


def _monomial_by_chebyshev(n):
    # https://en.wikipedia.org/wiki/Chebyshev_polynomials#Explicit_expressions
    cxk_to_ckx = np.zeros((n, n), dtype=complex)  # cos x^k = Σj c[k, j] Tj(cos x) = Σj c[k, j] cos jx
    for k in range(n):
        # naive_cheby = chebyt(k).c
        for j in range(k + 1):
            if j % 2 != k % 2:
                continue
            # Evaluate coefficient
            c = 2.0 if j != 0 else 1.0
            for l in range(1, (k - j) // 2 + 1):
                c *= (0.5 + (k + j) / (4 * l))
            c *= 2 ** (-(j + k) / 2)
            cxk_to_ckx[k, j] = c  # * naive_cheby[j]
    return cxk_to_ckx


def _test_monomial_by_chebyshev(n):
    cxk_to_ckx = _monomial_by_chebyshev(n)
    check_monomial = np.eye(n)  # jth coefficient of x^k = δ_{jk}
    test_monomial = np.zeros((n, n), dtype=complex)
    cheby_coeff_mat = np.array([np.pad(chebyt(j).c[::-1], (0, n - j - 1), 'constant') for j in range(n)])
    for k in range(n):
        for j in range(k + 1):
            test_monomial[k] += cxk_to_ckx[k, j] * cheby_coeff_mat[j, :]
    assert np.allclose(test_monomial, check_monomial)


def skew_square_filter(n, delta_t, omega):
    # Test required
    if not n % 2 == 1:
        raise ValueError
    shift_omega = -omega
    omega = 0.0
    vec = np.zeros(n, dtype=complex)
    n_half = n // 2
    for i, k in enumerate(range(-n_half, n_half + 1)):
        if k == 0:
            vec[i] = delta_t * shift_omega / (2 * np.pi) + 0.5
        else:
            x_k = (1j * np.exp(-1j * k * delta_t * shift_omega) - 1j * (-1) ** k) / (2 * np.pi * k)
            vec[i] = np.exp(1j * k * delta_t * omega) * x_k
    return vec


def filter_draw(delta_t, delta_e, n, filter_func_list):
    n_points = 1000
    delta_omega = 0.0
    omega_space = np.linspace(-2 * np.pi / delta_t, 2 * np.pi / delta_t, n_points)

    # filter_func = partial(square_filter, n, delta_t)
    # shift_omega = 0.0 # -np.pi / (2*delta_t)
    # filter_func = partial(skew_square_filter, n, delta_t, shift_omega)

    print(delta_t, delta_e, n)
    x = single_tone(0.0, n, delta_t, delta_omega) * n
    for filter_func_name, filter_func in filter_func_list.items():
        inner_product_outscope = list()
        inner_product = np.zeros(n_points, dtype=complex)
        print(filter_func_name)
        y = filter_func(0.0, n, delta_t, delta_e)
        print(f"\tomega=0.0: {np.abs(x.T.conj() @ y) ** 2}")
        for i, omega in enumerate(omega_space):
            x = single_tone(delta_omega, n, delta_t) * n
            y = filter_func(omega, n, delta_t, delta_e)
            inner_product[i] = np.abs(x.T.conj() @ y) ** 2
            if delta_e < np.abs(omega) < 2 * np.pi / delta_t - delta_e:
                inner_product_outscope.append(inner_product[i])
        print(f"\tmax_out: {np.max(inner_product_outscope)}")
        plt.plot(omega_space * (delta_t / np.pi), inner_product.real, label=filter_func_name)
    plt.axhline(0, color='red')
    plt.axhline(1, color='red')
    plt.axvline(delta_e * delta_t, color='red')
    plt.axvline(-delta_e * delta_t, color='red')
    plt.axvline(2.0 - delta_e * delta_t, color='red')
    plt.axvline(-2.0 + delta_e * delta_t, color='red')
    # plt.axvline(np.pi - delta_e * delta_t, color='red')
    # plt.axvline(- np.pi + delta_e * delta_t, color='red')
    plt.yscale('log')
    plt.ylim([1e-13, 1e1])
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # _test_monomial_by_chebyshev(10)
    delta_e = 0.1
    ham_norm = 1.0
    n = 31
    delta_t = np.pi / ham_norm
    fncs = {
            "single_tone": single_tone,
            "gaussian": gaussian_filter,
            "t_cheby": trigonometric_chebyshev_filter,
            }
    filter_draw(delta_t, delta_e, n, fncs)
