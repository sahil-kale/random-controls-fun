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

# Define second derivatives as symbols (not functions!)
xdd = sp.Symbol('xdd', real=True)
theta1dd = sp.Symbol('theta1dd', real=True)

# Replace second derivatives in equations
xdd_expr = sp.diff(x, t, t)
theta1dd_expr = sp.diff(theta_1, t, t)

eq1_raw = sp.diff(sp.diff(L, sp.diff(x, t)), t) - sp.diff(L, x) - F_a
eq2_raw = sp.diff(sp.diff(L, sp.diff(theta_1, t)), t) - sp.diff(L, theta_1)

eq1 = eq1_raw.subs({xdd_expr: xdd, theta1dd_expr: theta1dd})
eq2 = eq2_raw.subs({xdd_expr: xdd, theta1dd_expr: theta1dd})

# Optionally simplify here
eq1 = sp.simplify(eq1)
eq2 = sp.simplify(eq2)

# Solve
sol = sp.solve([eq1, eq2], (xdd, theta1dd), simplify=True, rational=False)
print("Equations of motion:")
sp.pprint(sol[xdd], use_unicode=True)
sp.pprint(sol[theta1dd], use_unicode=True)

print("Verifying the solution...")

# Substitute the solved expressions back into the raw equations
eq1_check = eq1_raw.subs({xdd_expr: sol[xdd], theta1dd_expr: sol[theta1dd]})
eq2_check = eq2_raw.subs({xdd_expr: sol[xdd], theta1dd_expr: sol[theta1dd]})

# Simplify the results to verify they reduce to 0
eq1_check_simplified = sp.simplify(eq1_check)
eq2_check_simplified = sp.simplify(eq2_check)

# Print to confirm
print("Residual of x equation after substitution:")
sp.pprint(eq1_check_simplified, use_unicode=True)

print("\nResidual of theta_1 equation after substitution:")
sp.pprint(eq2_check_simplified, use_unicode=True)