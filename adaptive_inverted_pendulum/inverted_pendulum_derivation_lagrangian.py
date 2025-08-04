import sympy as sp

# Symbol defs
t = sp.symbols('t', real=True)
g = sp.Symbol('g', real=True)  # gravitational acceleration
M_cart = sp.Symbol('M_cart', real=True)
F_a = sp.Function('F_a')(t)  # applied force on the cart

# link specific defs
l_1 = sp.Symbol('l_1', real=True)
m_1 = sp.Symbol('m_1', real=True)

# state variables
x = sp.Function('x')(t)  # cart position
theta_1 = sp.Function('theta_1')(t)  # angle of link 1

# Lagrangian: T - V, where T is kinetic energy and V is potential energy
T_cart = 0.5 * M_cart * sp.diff(x, t)**2
V_cart = 0 # doesn't count for anything

# Link 1 
pos_link_1_x = x + l_1 * sp.sin(theta_1)
vel_link_1_x = sp.diff(pos_link_1_x, t)
pos_link_1_y = -l_1 * sp.cos(theta_1)
vel_link_1_y = sp.diff(pos_link_1_y, t)
vel_link_1_squared = vel_link_1_x**2 + vel_link_1_y**2
T_link_1 = 0.5 * m_1 * vel_link_1_squared
V_link_1 = m_1 * g * pos_link_1_y

L = T_cart + T_link_1 - V_cart - V_link_1

# Equations of motion using Euler-Lagrange equations
# d/dt(dL/dq_dot) - dL/dq = Q_i
x_dot = sp.diff(x, t)
x_eq = sp.simplify(sp.diff(sp.diff(L, x_dot), t) - sp.diff(L, x)) - F_a # Equalling 0
theta_1_eq = sp.simplify(sp.diff(sp.diff(L, sp.diff(theta_1, t)), t) - sp.diff(L, theta_1)) # Equalling 0

sp.pprint(x_eq, use_unicode=True)
sp.pprint(theta_1_eq, use_unicode=True)