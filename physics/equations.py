import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from functools import lru_cache


def derive_double_pendulum_dynamics():
    """
        Assume two rigid rods with uniformly distributed mass connected in series.
        Returns the mass matrix M and the rest vector C such that:
            Returns M(q) and rest(q, qdot) such that M*qdd + rest = 0
    """

    # Define symbols
    m1, m2, l1, l2, g = sp.symbols('m1 m2 l1 l2 g')
    t = sp.symbols('t')
    th1 = sp.Function('th1')(t)
    th2 = sp.Function('th2')(t)
    th1_dot, th2_dot = sp.symbols('th1_dot th2_dot')
    th1_ddot, th2_ddot = sp.symbols('th1_ddot th2_ddot')

    substitutions = {
        sp.diff(th1, t): th1_dot,
        sp.diff(th2, t): th2_dot,
        sp.diff(th1, t, t): th1_ddot,
        sp.diff(th2, t, t): th2_ddot
    }

    # Define center of mass positions
    x1 = l1/2 * sp.sin(th1)
    y1 = -l1/2 * sp.cos(th1)
    x2 = l1 * sp.sin(th1) + l2/2 * sp.sin(th2)
    y2 = -l1 * sp.cos(th1) - l2/2 * sp.cos(th2)

    # Define velocities
    vx1 = sp.diff(x1, t)
    vy1 = sp.diff(y1, t)
    vx2 = sp.diff(x2, t)
    vy2 = sp.diff(y2, t)
    v1_sq = vx1**2 + vy1**2 # type: ignore
    v2_sq = vx2**2 + vy2**2 # type: ignore

    # Kinetic energy (translational + rotational about center of mass)
    K1 = sp.Rational(1, 2) * m1 * v1_sq + sp.Rational(1, 2) * (sp.Rational(1, 12) * m1 * l1**2) * sp.diff(th1, t)**2
    K2 = sp.Rational(1, 2) * m2 * v2_sq + sp.Rational(1, 2) * (sp.Rational(1, 12) * m2 * l2**2) * sp.diff(th2, t)**2
    K = K1 + K2

    # Potential energy
    U1 = m1 * g * y1
    U2 = m2 * g * y2
    U = U1 + U2

    # Lagrangian and equations of motion
    L = K - U
    dL_dtheta1 = sp.diff(L, th1)
    dL_dtheta2 = sp.diff(L, th2)
    dL_dtheta1_dot = sp.diff(L, sp.diff(th1, t))
    dL_dtheta2_dot = sp.diff(L, sp.diff(th2, t))
    ddt_dL_dtheta1_dot = sp.diff(dL_dtheta1_dot, t)
    ddt_dL_dtheta2_dot = sp.diff(dL_dtheta2_dot, t)

    R = sp.Matrix(
        [ddt_dL_dtheta1_dot - dL_dtheta1, # type: ignore
        ddt_dL_dtheta2_dot - dL_dtheta2] # type: ignore
    )
    R_sub = R.subs(substitutions).expand()

    qdd = [th1_ddot, th2_ddot]
    M, rest = sp.linear_eq_to_matrix(R_sub, qdd)
    M.simplify()
    rest.simplify()

    symbols = {
        'theta1': th1,
        'theta2': th2,
        'theta1_dot': th1_dot,
        'theta2_dot': th2_dot,
        'parameters': (m1, m2, l1, l2, g)
    }

    return M, rest, symbols

@lru_cache(maxsize=1)
def build_numpy_functions():
    """
    Build lambdified functions for M(q) and C(q, qdot) using NumPy.
    Used for fast numerical simulation with scipy.integrate.
    """
    print("Deriving equations symbolically (this may take a moment)...")
    M_sym, C_sym, sym = derive_double_pendulum_dynamics()
    
    th1, th2 = sym["theta1"], sym["theta2"]
    th1_d, th2_d = sym["theta1_dot"], sym["theta2_dot"]
    m1, m2, l1, l2, g = sym["parameters"]
    
    # Lambdify for NumPy
    M_func = sp.lambdify(
        (th1, th2, m1, m2, l1, l2),
        M_sym,
        modules="numpy"
    )
    
    C_func = sp.lambdify(
        (th1, th2, th1_d, th2_d, m1, m2, l1, l2, g),
        C_sym,
        modules="numpy"
    )
    
    print("Symbolic derivation complete!")
    return M_func, C_func


# Globals to hold the built numpy functions
_M_func_numpy, _C_func_numpy = None, None


def _ensure_numpy_built():
    global _M_func_numpy, _C_func_numpy
    if _M_func_numpy is None:
        _M_func_numpy, _C_func_numpy = build_numpy_functions()


def double_pendulum_derivatives(t, y, params):
    _ensure_numpy_built()
    
    theta1, theta2, omega1, omega2 = y
    m1, m2, l1, l2, g = params

    M = _M_func_numpy(theta1, theta2, m1, m2, l1, l2)
    rest = _C_func_numpy(theta1, theta2, omega1, omega2, m1, m2, l1, l2, g)

    # Ensure rest is a 1D array
    rest = np.array(rest).flatten()
    
    # Đụ má cái này để rest chứ không phải trừ rest nguyên một ngày của tao quá mệt rồi
    gamma = np.linalg.solve(M, rest)

    return np.array([omega1, omega2, gamma[0], gamma[1]])


def compute_energy(q, qdot, m1, m2, l1, l2, g):
    """
    Compute total energy for verification (uniform rods).
    Matches the Lagrangian used in derive_double_pendulum_dynamics.
    """
    theta1, theta2 = q[:, 0], q[:, 1]
    omega1, omega2 = qdot[:, 0], qdot[:, 1]
    
    # Center of mass positions
    y1 = -l1/2 * np.cos(theta1)
    y2 = -l1 * np.cos(theta1) - l2/2 * np.cos(theta2)
    
    # Potential energy
    U = m1 * g * y1 + m2 * g * y2
    
    # Velocities of centers of mass
    # v_cm = d/dt(position), using chain rule: d/dt = omega * d/dtheta
    vx1 = l1/2 * omega1 * np.cos(theta1)
    vy1 = l1/2 * omega1 * np.sin(theta1)
    
    vx2 = l1 * omega1 * np.cos(theta1) + l2/2 * omega2 * np.cos(theta2)
    vy2 = l1 * omega1 * np.sin(theta1) + l2/2 * omega2 * np.sin(theta2)
    
    # Kinetic energy: translational + rotational
    K_trans_1 = 0.5 * m1 * (vx1**2 + vy1**2)
    K_rot_1 = 0.5 * (1/12 * m1 * l1**2) * omega1**2  # (1/2) * I * omega^2
    
    K_trans_2 = 0.5 * m2 * (vx2**2 + vy2**2)
    K_rot_2 = 0.5 * (1/12 * m2 * l2**2) * omega2**2
    
    K = K_trans_1 + K_rot_1 + K_trans_2 + K_rot_2
    
    return K + U

@lru_cache(maxsize=1)
def build_torch_functions():
    """
    Build lambdified functions for M(q) and C(q, qdot).
    Returns M_fn, C_fn that work with PyTorch tensors.
    """
    import torch  # Import here to ensure it's available
    M_sym, C_sym, sym = derive_double_pendulum_dynamics()
    th1, th2 = sym["theta1"], sym["theta2"]
    th1_d, th2_d = sym["theta1_dot"], sym["theta2_dot"]
    m1, m2, l1, l2, g = sym["parameters"]

    # Lambdify each element of the matrix/vector separately
    # M is (2, 2) matrix
    M_elements = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(sp.lambdify((th1, th2, m1, m2, l1, l2), 
                                   M_sym[i, j], 
                                   modules=torch))
        M_elements.append(row)
    
    # C is (2, 1) vector
    C_elements = []
    for i in range(2):
        C_elements.append(sp.lambdify((th1, th2, th1_d, th2_d, m1, m2, l1, l2, g), 
                                      C_sym[i], 
                                      modules=torch))

    def M_fn(theta1, theta2, m1, m2, l1, l2):
        """Returns M matrix as torch tensor (N, 2, 2) or (2, 2)"""
        import torch
        
        # Determine if we're in batch mode
        is_batch = isinstance(theta1, torch.Tensor) and theta1.ndim > 0
        
        if is_batch:
            # Ensure parameters are broadcast to batch size
            batch_size = theta1.shape[0]
            device = theta1.device
            dtype = theta1.dtype
            
            # Expand scalar parameters to match batch
            if isinstance(m1, torch.Tensor) and m1.ndim == 0:
                m1_exp = m1.expand(batch_size)
                m2_exp = m2.expand(batch_size)
                l1_exp = l1.expand(batch_size)
                l2_exp = l2.expand(batch_size)
            else:
                m1_exp, m2_exp, l1_exp, l2_exp = m1, m2, l1, l2
        else:
            m1_exp, m2_exp, l1_exp, l2_exp = m1, m2, l1, l2
        
        # Compute each element
        M00 = M_elements[0][0](theta1, theta2, m1_exp, m2_exp, l1_exp, l2_exp)
        M01 = M_elements[0][1](theta1, theta2, m1_exp, m2_exp, l1_exp, l2_exp)
        M10 = M_elements[1][0](theta1, theta2, m1_exp, m2_exp, l1_exp, l2_exp)
        M11 = M_elements[1][1](theta1, theta2, m1_exp, m2_exp, l1_exp, l2_exp)
        
        # Convert to tensors if they aren't already
        if not isinstance(M00, torch.Tensor):
            M00 = torch.tensor(M00, dtype=torch.float32)
            M01 = torch.tensor(M01, dtype=torch.float32)
            M10 = torch.tensor(M10, dtype=torch.float32)
            M11 = torch.tensor(M11, dtype=torch.float32)
        
        # Stack into matrix
        if is_batch:
            N = theta1.shape[0]
            device = theta1.device
            dtype = theta1.dtype
            M = torch.zeros(N, 2, 2, dtype=dtype, device=device)
            M[:, 0, 0] = M00
            M[:, 0, 1] = M01
            M[:, 1, 0] = M10
            M[:, 1, 1] = M11
        else:
            # Single point mode
            M = torch.stack([
                torch.stack([M00, M01]),
                torch.stack([M10, M11])
            ])
        
        return M

    def C_fn(theta1, theta2, theta1_dot, theta2_dot, m1, m2, l1, l2, g):
        """Returns C vector as torch tensor (N, 2) or (2,)"""
        # Determine if we're in batch mode
        is_batch = isinstance(theta1, torch.Tensor) and theta1.ndim > 0
        
        if is_batch:
            # Ensure parameters are broadcast to batch size
            batch_size = theta1.shape[0]
            
            # Expand scalar parameters to match batch
            if isinstance(m1, torch.Tensor) and m1.ndim == 0:
                m1_exp = m1.expand(batch_size)
                m2_exp = m2.expand(batch_size)
                l1_exp = l1.expand(batch_size)
                l2_exp = l2.expand(batch_size)
                g_exp = g.expand(batch_size)
            else:
                m1_exp, m2_exp, l1_exp, l2_exp, g_exp = m1, m2, l1, l2, g
        else:
            m1_exp, m2_exp, l1_exp, l2_exp, g_exp = m1, m2, l1, l2, g
        
        # Compute each element
        C0 = C_elements[0](theta1, theta2, theta1_dot, theta2_dot, m1_exp, m2_exp, l1_exp, l2_exp, g_exp)
        C1 = C_elements[1](theta1, theta2, theta1_dot, theta2_dot, m1_exp, m2_exp, l1_exp, l2_exp, g_exp)
        
        # Convert to tensors if they aren't already
        if not isinstance(C0, torch.Tensor):
            C0 = torch.tensor(C0, dtype=torch.float32)
            C1 = torch.tensor(C1, dtype=torch.float32)
        
        # Stack into vector
        if is_batch:
            C = torch.stack([C0, C1], dim=1)  # (N, 2)
        else:
            C = torch.stack([C0, C1])
        
        return C

    return M_fn, C_fn

# Globals to hold the built functions
_M_fn_torch, _C_fn_torch = None, None

def _ensure_built():
    global _M_fn_torch, _C_fn_torch
    if _M_fn_torch is None:
        _M_fn_torch, _C_fn_torch = build_torch_functions()

def get_M_fn():
    _ensure_built()
    return M_fn

def get_C_fn():
    _ensure_built()
    return C_fn

if __name__ == "__main__":
    # Test the derivation
    M, rest, symbols = derive_double_pendulum_dynamics()
    print("Mass Matrix M:")
    sp.pprint(M)
    print("\nRest Vector C:")
    sp.pprint(rest)