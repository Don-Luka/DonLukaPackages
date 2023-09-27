import numpy as np

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

def solve_IVP(t_init, t_fin, uv_0, K_, C_, M_, q_, qval, method_used = 'Radau', r_tol=1e-3 ,a_tol=1e-6):
    def f_u(v_):
        return v_
    def f_v(t_, u_, v_, K_, C_, M_, q_):
        return np.linalg.inv(M_)@ (q_* qval(t_)-K_@ u_-C_@ v_)
    def deriv(t_, x_, K_, C_, M_, q_):
        u = x_[::2]
        v = x_[1::2]
        dxdt = np.zeros_like(x_)
        dudt = dxdt[::2]
        dvdt = dxdt[1::2]
        dudt[:] = f_u(v[:])
        dvdt[:] = f_v(t_, u[:], v[:], K_, C_, M_, q_)
        return dxdt
    sol0 = solve_ivp(deriv, [t_init,t_fin], uv_0, method = method_used, args = (K_, C_, M_, q_), rtol=r_tol ,atol=a_tol)
    return (sol0.y).T, sol0.t

def solve_IVP_odeint(uv_0, t_, K_, C_, M_, q_, qval, dt_max):
    def f_u(v_):
        return v_
    def f_v(u_, v_, t_, K_, C_, M_, q_):
        return np.linalg.inv(M_)@ (q_* qval(t_)-K_@ u_-C_@ v_)
    def deriv(x_, t_, K_, C_, M_, q_):
        u = x_[::2]
        v = x_[1::2]
        dxdt = np.zeros_like(x_)
        dudt = dxdt[::2]
        dvdt = dxdt[1::2]
        dudt[:] = f_u(v[:])
        dvdt[:] = f_v(u[:], v[:], t_, K_, C_, M_, q_)
        return dxdt
    sol0 = odeint(deriv, uv_0, t_, args = (K_, C_, M_, q_), full_output=0, hmax=dt_max)
    return sol0