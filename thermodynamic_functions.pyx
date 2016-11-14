include "parameters.pxi"
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, fabs,atan, exp, fmax, pow




# - th_l(T, p, qt) ok
# - ql, T = f(th_l, qt, p)
# - T(th_l, p, qt)


'''Density'''
cpdef double alpha_from_tp(double p0, double T, double  qt, double qv):
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv)
# cpdef inline double alpha(const double p0, const double T, const double qt, const double qv) nogil:
#     return alpha_c(p0, T, qt, qv)


'''Pressure'''
cpdef double pd_c(const double p0, const double qt, const double qv) nogil:
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv)
#     # return pd_c(p0, qt, qv)

cpdef double pv_c(double p0, double qt, double qv):
    a = p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv)
    # print('pv_c: ', p0, qt, qv, a)
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv)


'''Potential Temperature'''
cpdef double exner(const double p):
    return pow((p/p_tilde),kappa)

# // Dry potential temperature
cpdef double theta(const double p, const double T):
    return T / exner(p)

# // Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
cpdef double thetali(const double p, const double T, const double qt, const double ql, const double qi):
    L = latent_heat(T)
    return theta(p, T) * exp(-L*(ql/(1.0 - qt) + qi/(1.0 -qt))/(T*cpd))

# // Virtual potential temperature
cpdef double thetav(const double p, const double T, const double qt):
    return theta(p,T) * (1 + (Rv/Rd - 1)*qt)

# # // Entropy potential temperature
# cpdef inline double thetas(const double s, const double qt) nogil:
#     return T_tilde*exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt))
#     # return thetas_c(s, qt)




'''Entropies'''
cpdef double sd_c(double pd, double T):
    return sd_tilde + cpd*np.log(T/T_tilde) -Rd*np.log(pd/p_tilde)

cpdef double sv_c(double pv, double T):
    # !! Problem: sv_c(pv, T) is ill-defined for qv=0, since pv_c(p0, qt, qv)~log(pv)=inf
    return sv_tilde + cpv*np.log(T/T_tilde) - Rv * np.log(pv/p_tilde)

cpdef double sc_c(double L, double T):
    return -L/T

cpdef double entropy_from_tp(double p0, double T, double qt, double ql, double qi):
    cdef:
        double qv = qt - ql - qi
        double qd = 1.0 - qt
        double pd = pd_c(p0, qt, qv)
        double pv = pv_c(p0, qt, qv)
        # double Lambda = self.Lambda_fp(T)
        # double L = self.L_fp(T, Lambda)
        double L = latent_heat(T)
        double ret = 0.0
    # !! Problem: sv_c(pv, T) is ill-defined for qv=0, since pv_c(p0, qt, qv)~log(pv)=inf
    if qt == 0:
        ret = sd_c(pd, T)
    else:
        ret = sd_c(pd, T) * (1.0 - qt) + sv_c(pv, T) * qt + sc_c(L, T) * (ql + qi)
    return ret



'''Clausius Clapeyron: Latent Heat, Saturation Pressure'''
cpdef double latent_heat(double T):
    cdef double TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC * TC - 0.00006 * TC * TC * TC) * 1000.0

cpdef double pv_star(double T)   :
    #    Magnus formula
    cdef double TC = T - 273.15
    return 6.1094*exp((17.625*TC)/float(TC+243.04))*100

cpdef double qv_star_c(const double p0, const double qt, const double pv):
    return eps_v * (1.0 - qt) * pv / (p0 - pv)
    # return qv_star_c(p0, qt, pv)



'''Saturation Adjustment'''
cpdef double eos_first_guess_thetal(double s, double pd, double pv, double qt)   :
    cdef double p0 = pd + pv
    return s * exner(p0)

cpdef double eos_first_guess_entropy(double s, double pd, double pv, double qt )   :
    ## PROBLEM: nan in a, b=inf
    cdef double qd = 1.0 - qt
    a = (T_tilde * exp((s - qd*(sd_tilde - Rd *log(pd/p_tilde))
                              - qt * (sv_tilde - Rv * log(pv/p_tilde)))/((qd*cpd + qt * cpv))))
    b = T_tilde * exp(s - qd*(sd_tilde - Rd *log(pd/p_tilde)))
    c = qd*(sd_tilde - Rd *log(pd/p_tilde))
    # if np.isnan(a):
        # print('')
        # print('eos first guess: ', a, b, c, s)
        # a = nan if s small (s=0.005), qt = 6900 ~ -qv
        # ???? qt = 6900 ??? qv < 0 ???
        # print('qt', qt, 'qd', qd)
        # print(qt, qd, s, T_tilde, sd_tilde, Rd, Rv, pv, p_tilde)
        # print(log(pv/p_tilde), cpd, cpv)
        # print('')
        # print('')
    return (T_tilde *exp((s - qd*(sd_tilde - Rd *log(pd/p_tilde))
                              - qt * (sv_tilde - Rv * log(pv/p_tilde)))/((qd*cpd + qt * cpv))))

# cpdef eos_struct eos( t_to_prog, prog_to_t,double p0, double qt, double prog):
cpdef eos_struct eos(double p0, double qt, double prog):
# cpdef eos(double p0, double qt, double prog):
    cdef double qv = qt
    # cdef double ql = 0.0
    # cdef double T = 0.0
    cdef eos_struct _ret    # ql = _ret['ql'], T = _ret['T']

    cdef double pv_1 = pv_c(p0,qt,qt)
    cdef double pd_1 = p0 - pv_1
    # cdef double T_1 = prog_to_t(prog, pd_1, pv_1, qt)
    cdef double T_1 = eos_first_guess_entropy(prog, pd_1, pv_1, qt)

    cdef double pv_star_1 = pv_star(T_1)
    cdef double qv_star_1 = qv_star_c(p0,qt,pv_star_1)

    cdef double ql_1, prog_1, f_1, T_2, delta_T
    cdef double qv_star_2, ql_2=0.0, pv_star_2, pv_2, pd_2, prog_2, f_2
    # If not saturated

    # if np.isnan(T_1):
        # print('T_1', T_1, eos_first_guess_entropy(prog, pd_1, pv_1, qt), prog, pd_1, pv_1, qt)
        # print('pd_q', pd_1, 'pv_1', pv_1, p0, qt)
        # ???? pv_1 > 250'000 --> pd_q < 0
    if(qt <= qv_star_1):
        # print('unsaturated')
        # T = T_1
        # ql = 0.0
        _ret.T = T_1
        _ret.ql = 0.0

    else:
        ql_1 = qt - qv_star_1
        prog_1 = entropy_from_tp(p0, T_1, qt, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt)*cpd + qv_star_1 * cpv)
        delta_T  = fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt,pv_star_2)
            pv_2 = pv_c(p0, qt, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt - qv_star_2
            prog_2 =  entropy_from_tp(p0, T_2, qt, ql_2, 0.0)
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = fabs(T_2 - T_1)

        _ret.T  = T_2
        # T = T_2
        qv = qv_star_2
        _ret.ql = ql_2
        # ql = ql_2

    return _ret
    # return T, ql













# cpdef inline double density_temperature(const double T, const double qt, const double qv) nogil:
#     return density_temperature_c(T, qt, qv)
#
# cpdef inline double theta_rho(const double p0, const double T, const double qt, const double qv) nogil:
#     return theta_rho_c(p0, T, qt, qv)
#
# cpdef inline double cpm(const double qt) nogil:
#     return cpm_c(qt)
#
# cpdef inline double thetas_t(const double p0, const double T, const double qt, const double qv,
#                       const double qc, const double L) nogil:
#     return thetas_t_c( p0,  T, qt, qv, qc, L)
#
# cpdef inline double entropy_from_thetas(const double thetas, const double qt) nogil:
#     return cpm_c(qt) * log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde
#     # return entropy_from_thetas_c(thetas, qt)
#
# cpdef inline double buoyancy(const double alpha0, const double alpha) nogil:
#     return buoyancy_c(alpha0, alpha)


