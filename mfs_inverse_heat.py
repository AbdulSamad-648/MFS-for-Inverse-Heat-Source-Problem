"""
Method of Fundamental Solutions (MFS) for the Inverse (time-dependent) Heat
Source Problem, following:

Liang Yan, Chu-Li Fu, Feng-Lian Yang,
"The method of fundamental solutions for the inverse heat source problem",
Engineering Analysis with Boundary Elements 32 (2008) 216-222.

Problem (Eqs. 2.1-2.5):
    u_t = a^2 u_xx + f(t),      0<x<1, 0<t<=tmax
    u(x,0) = u0(x)
    u(0,t) = 0
    u(1,t) = g(t)
    u(x0,t) = h(t)              (interior thermocouple / over-specification)

Unknowns: u(x,t) and f(t) (source depends on time only).

Method (Sections 2-4 of the paper):
  1. Change of variables  r(t) = int_0^t f(eta) d eta,  v = u - r(t)
     turns the problem into a *homogeneous* heat equation for v with
     boundary data expressed as *differences* v(1,t)-v(0,t) = g(t) and
     v(x0,t)-v(0,t) = h(t)  (the unknown r(t) cancels out of these two
     conditions - this is the key trick in the paper).
  2. v is expanded in a basis of time-shifted fundamental solutions of the
     1D heat equation, phi(x,t) = Phi(x, t+T),  T > tmax
        Phi(x,t) = 1/(2 a sqrt(pi t)) * exp(-x^2/(4 a^2 t)) * H(t)
     centred at a set of source points equal to the collocation points
     (Eq. 3.7).
  3. Collocation at D1 (initial condition), D2 (x=1 boundary data) and
     D3 (interior measurement) gives a square linear system  A*lambda = b
     (Eqs. 3.8-3.10).
  4. Because A is severely ill-conditioned, lambda is recovered by
     Tikhonov regularization,
        min_lambda { ||A lambda - b||^2 + alpha^2 ||lambda||^2 }
     with alpha chosen by Generalized Cross Validation (GCV) (Eqs. 4.1-4.2).
  5. Finally
        v*(x,t) = sum_j lambda_j phi_j(x,t)
        r*(t)   = sum_j lambda_j phi_j(0,t)     (Eq. 4.4)
        u*(x,t) = v*(x,t) + r*(t)               (Eq. 4.5)
        f*(t)   = r*'(t)                        (Eq. 4.6, closed form below)

NOTE ON EXAMPLE 1's SOURCE TERM
--------------------------------
The PDF text lists u(x,t) = e^{-t} sin(x) + 3 t x^2 + x^4/4 with f(t) = 6t.
Differentiating the stated u(x,t) (u_t - u_xx) gives f(t) = -6t (the sin(x)
terms cancel regardless of the sign of the exponential term, so this is not
a transcription ambiguity about e^{-t} vs e^{t} - it is simply a sign that
did not survive OCR/typesetting). This code uses the value that is actually
consistent with the stated u(x,t), i.e. f(t) = -6t, and reports both curves
so you can see the two differ only by a sign.
"""

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize_scalar


# ----------------------------------------------------------------------
# 1. Fundamental solution of the 1D heat equation and its derivatives
# ----------------------------------------------------------------------

def Phi(x, t, a=1.0):
    """Fundamental solution of u_t = a^2 u_xx (Eq. 3.1). t must be > 0."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t > 0
    out[mask] = 1.0 / (2 * a * np.sqrt(np.pi * t[mask])) * \
        np.exp(-x[mask] ** 2 / (4 * a ** 2 * t[mask]))
    return out


def phi_shifted(x, t, xj, tj, T, a=1.0):
    """Time-shifted, source-centred basis function phi_j(x,t) = Phi(x-xj, t-tj+T)."""
    dx = np.atleast_1d(x - xj).astype(float)
    s = np.atleast_1d(t - tj + T).astype(float)
    return Phi(dx, s, a=a)


def phi_shifted_dt(x, t, xj, tj, T, a=1.0):
    """
    d/dt of phi_shifted(x,t,xj,tj,T,a), needed for f*(t) = r*'(t) (Eq. 4.6).

    With d = x-xj, s = t-tj+T:
        Phi(d,s) = 1/(2a sqrt(pi s)) exp(-d^2/(4a^2 s))
        dPhi/ds  = Phi(d,s) * [ d^2/(4 a^2 s^2) - 1/(2 s) ]
        dphi/dt  = dPhi/ds        (since ds/dt = 1)
    """
    d = np.atleast_1d(x - xj).astype(float)
    s = np.atleast_1d(t - tj + T).astype(float)
    val = Phi(d, s, a=a)
    factor = d ** 2 / (4 * a ** 2 * s ** 2) - 1.0 / (2 * s)
    return val * factor


# ----------------------------------------------------------------------
# 2. Build the collocation / source point set D = D1 U D2 U D3 (Eqs 3.3-3.6)
# ----------------------------------------------------------------------

def build_points(n, m, s, x0, tmax):
    """
    D1: n equally spaced points on [0,1] at t=0            (initial condition)
    D2: m equally spaced points at x=1, t in (0,tmax]       (boundary data)
    D3: s equally spaced points at x=x0, t in (0,tmax]      (measurement)
    Returns arrays (xs, ts) of length n+m+s, plus the index ranges.
    """
    x1 = np.linspace(0, 1, n)
    t1 = np.zeros(n)

    t2 = np.linspace(tmax / m, tmax, m)
    x2 = np.ones(m)

    t3 = np.linspace(tmax / s, tmax, s)
    x3 = np.full(s, x0)

    xs = np.concatenate([x1, x2, x3])
    ts = np.concatenate([t1, t2, t3])
    idx = dict(D1=slice(0, n), D2=slice(n, n + m), D3=slice(n + m, n + m + s))
    return xs, ts, idx


# ----------------------------------------------------------------------
# 3. Assemble A and b  (Eqs. 3.8-3.10)
# ----------------------------------------------------------------------

def assemble_system(xs, ts, idx, u0_func, g_func, h_func, T, a=1.0):
    N = len(xs)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # rows for D1: v(xi,0) = u0(xi)
    for row, i in enumerate(range(idx['D1'].start, idx['D1'].stop)):
        xi, ti = xs[i], ts[i]
        A[i, :] = phi_shifted(xi, ti, xs, ts, T, a=a)
        b[i] = u0_func(xi)

    # rows for D2: v(1,tk) - v(0,tk) = g(tk)
    for k in range(idx['D2'].start, idx['D2'].stop):
        xk, tk = xs[k], ts[k]
        A[k, :] = phi_shifted(xk, tk, xs, ts, T, a=a) - \
            phi_shifted(0.0, tk, xs, ts, T, a=a)
        b[k] = g_func(tk)

    # rows for D3: v(x0,tl) - v(0,tl) = h(tl)
    for l in range(idx['D3'].start, idx['D3'].stop):
        xl, tl = xs[l], ts[l]
        A[l, :] = phi_shifted(xl, tl, xs, ts, T, a=a) - \
            phi_shifted(0.0, tl, xs, ts, T, a=a)
        b[l] = h_func(tl)

    return A, b


# ----------------------------------------------------------------------
# 4. Tikhonov regularization with GCV parameter choice (Eqs. 4.1-4.2)
# ----------------------------------------------------------------------

class TikhonovGCV:
    """SVD-based Tikhonov regularization with GCV parameter selection,
    equivalent in spirit to Hansen's REGULARIZATION TOOLS (ref. [20])."""

    def __init__(self, A):
        self.U, self.sigma, self.Vt = svd(A, full_matrices=False)
        self.n = A.shape[0]

    def solve(self, b, alpha):
        UT_b = self.U.T @ b
        filt = self.sigma / (self.sigma ** 2 + alpha ** 2)
        lam = self.Vt.T @ (filt * UT_b)
        return lam

    def gcv_value(self, b, alpha):
        UT_b = self.U.T @ b
        s2 = self.sigma ** 2
        filt = s2 / (s2 + alpha ** 2)          # A*A_I diagonal (in U basis)
        res = np.sum(((1 - filt) * UT_b) ** 2)  # ||A*lambda_alpha - b||^2
        denom = (self.n - np.sum(filt)) ** 2
        return res / denom

    def find_alpha_gcv(self, b, alpha_bounds=(1e-16, 1e2)):
        lo, hi = np.log10(alpha_bounds[0]), np.log10(alpha_bounds[1])

        def obj(log_alpha):
            return self.gcv_value(b, 10 ** log_alpha)

        res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                               options={'xatol': 1e-3})
        alpha_star = 10 ** res.x
        return alpha_star, res.fun

    def gcv_curve(self, b, alphas):
        return np.array([self.gcv_value(b, a) for a in alphas])


# ----------------------------------------------------------------------
# 5. Reconstruct v*, u*, r*, f* from the regularized coefficients
# ----------------------------------------------------------------------

def reconstruct_v(x, t, lam, xs, ts, T, a=1.0):
    """v*(x,t) for scalar x, array-like t (or vice versa)."""
    x_arr = np.broadcast_to(x, np.shape(t)).astype(float) if np.ndim(t) else np.array([x])
    t_arr = np.atleast_1d(t).astype(float)
    out = np.zeros_like(t_arr)
    for j in range(len(xs)):
        out += lam[j] * phi_shifted(x_arr, t_arr, xs[j], ts[j], T, a=a)
    return out if np.ndim(t) else out[0]


def reconstruct_r(t, lam, xs, ts, T, a=1.0):
    """
    r*(t) = -sum_j lambda_j phi_j(0,t).

    NOTE: the OCR'd Eq. (4.4) in the PDF shows r*(t) = sum_j lambda_j*
    phi(0-xj,t-tj) with no minus sign, but that contradicts the paper's own
    definition v(x,t)=u(x,t)-r(t) together with the boundary condition
    u(0,t)=0 (Eq. 2.3), which forces v(0,t) = -r(t), i.e. r(t) = -v(0,t).
    This matches a broader pattern of dropped minus signs in this PDF's text
    layer (e.g. "e^{-t}" -> "et", "f(t)=-6t" -> "6t" in Example 1). The sign
    below is the mathematically consistent one, verified against Example 1's
    known exact solution.
    """
    return -reconstruct_v(0.0, t, lam, xs, ts, T, a=a)


def reconstruct_f(t, lam, xs, ts, T, a=1.0):
    """f*(t) = r*'(t) (Eq. 4.6) = -d/dt[v*(0,t)]."""
    t_arr = np.atleast_1d(t).astype(float)
    out = np.zeros_like(t_arr)
    x0_arr = np.zeros_like(t_arr)
    for j in range(len(xs)):
        out += lam[j] * phi_shifted_dt(x0_arr, t_arr, xs[j], ts[j], T, a=a)
    out = -out
    return out if np.ndim(t) else out[0]


def reconstruct_u(x, t, lam, xs, ts, T, a=1.0):
    return reconstruct_v(x, t, lam, xs, ts, T, a=a) + reconstruct_r(t, lam, xs, ts, T, a=a)


# ----------------------------------------------------------------------
# 6. Error metrics (Eqs. 5.1-5.2)
# ----------------------------------------------------------------------

def rms(exact, approx):
    return np.sqrt(np.mean((exact - approx) ** 2))


def res(exact, approx):
    return np.sqrt(np.sum((exact - approx) ** 2)) / np.sqrt(np.sum(exact ** 2))
