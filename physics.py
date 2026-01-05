import sympy as sp



def derive_double_pendulum_dynamics():
    """
        Assume two rigid rods with uniformly distributed mass connected in series.
        Returns the mass matrix M and the rest vector C such that:
            Returns M(q) and rest(q, qdot) such that M*qdd + rest = 0
    """

    # Define symbols
    m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g')
    t = sp.symbols('t')
    theta1 = sp.Function('theta1')(t)
    theta2 = sp.Function('theta2')(t)
    theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot')
    theta1_ddot, theta2_ddot = sp.symbols('theta1_ddot theta2_ddot')

    substitutions = {
        sp.diff(theta1, t): theta1_dot,
        sp.diff(theta2, t): theta2_dot,
        sp.diff(theta1, t, t): theta1_ddot,
        sp.diff(theta2, t, t): theta2_ddot
    }

    # Define center of mass positions
    x1 = l1/2 * sp.sin(theta1)
    y1 = -l1/2 * sp.cos(theta1)
    x2 = l1 * sp.sin(theta1) + l2/2 * sp.sin(theta2)
    y2 = -l1 * sp.cos(theta1) - l2/2 * sp.cos(theta2)

    # Define velocities
    vx1 = sp.diff(x1, t)
    vy1 = sp.diff(y1, t)
    vx2 = sp.diff(x2, t)
    vy2 = sp.diff(y2, t)
    v1_sq = vx1**2 + vy1**2 # type: ignore
    v2_sq = vx2**2 + vy2**2 # type: ignore

    # Kinetic energy
    K1 = sp.Rational(1, 2) * m1 * v1_sq + sp.Rational(1, 12) * m1 * l1**2 * sp.diff(theta1, t)**2
    K2 = sp.Rational(1, 2) * m2 * v2_sq + sp.Rational(1, 12) * m2 * l2**2 * sp.diff(theta2, t)**2
    K = K1 + K2

    # Potential energy
    U1 = m1 * g * y1
    U2 = m2 * g * y2
    U = U1 + U2

    # Lagrangian and equations of motion
    L = K - U
    dL_dtheta1 = sp.diff(L, theta1)
    dL_dtheta2 = sp.diff(L, theta2)
    dL_dtheta1_dot = sp.diff(L, sp.diff(theta1, t))
    dL_dtheta2_dot = sp.diff(L, sp.diff(theta2, t))
    ddt_dL_dtheta1_dot = sp.diff(dL_dtheta1_dot, t)
    ddt_dL_dtheta2_dot = sp.diff(dL_dtheta2_dot, t)

    R = sp.Matrix(
        [ddt_dL_dtheta1_dot - dL_dtheta1, # type: ignore
        ddt_dL_dtheta2_dot - dL_dtheta2] # type: ignore
    )
    R_sub = R.subs(substitutions).expand()

    qdd = [theta1_ddot, theta2_ddot]
    M, rest = sp.linear_eq_to_matrix(R_sub, qdd)
    M.simplify()
    rest.simplify()
    M = M.applyfunc(sp.nsimplify)
    rest = rest.applyfunc(sp.nsimplify)

    symbols = {
        'theta1': theta1,
        'theta2': theta2,
        'theta1_dot': theta1_dot,
        'theta2_dot': theta2_dot,
        'parameters': (m1, m2, l1, l2, g)
    }

    return M, rest, symbols