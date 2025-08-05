import numpy as np
import sympy as sp

class MultiLinkPendulum:
    def __init__(self, num_links, link_lengths, link_masses, cart_mass):
        self.num_links = num_links
        self.cart_mass = cart_mass
        self.link_lengths = link_lengths
        self.link_masses = link_masses

        assert len(link_lengths) == num_links, "Number of link lengths must match num_links"
        assert len(link_masses) == num_links, "Number of link masses must match num_links"
        self.derive_non_linear_state_space()


    def derive_non_linear_state_space(self):
        # Use the lagrangian method to derive the equations of motion
        # Where L = T - V, T is kinetic energy and V is potential energy
        # T for links is 0.5 * m_i * v_i^2
        # V for links is m_i * g * h_i, where h_i is the height of the link's center of mass relative to the ground

        t = sp.Symbol('t', real=True)
        g = sp.Symbol('g', real=True)  # gravitational acceleration
        M_cart = sp.Symbol('M_cart', real=True)
        F_a = sp.Function('F_a')(t)  # applied force on the cart

        cart_pos = sp.Function('x')(t)
        T_cart = 0.5 * M_cart * sp.diff(cart_pos, t)**2
        V_cart = 0 # doesn't count for anything
        
        link_angles = [sp.Function(f'theta_{i+1}')(t) for i in range(self.num_links)]
        link_com_positions = []
        link_kinetic_energies = []
        link_potential_energies = []

        for i in range(self.num_links):
            l_i = sp.Symbol(f'l_{i+1}', real=True)
            m_i = sp.Symbol(f'm_{i+1}', real=True)
            theta_i = link_angles[i]

            if i == 0:
                pos_link_x = cart_pos + l_i * sp.sin(theta_i)
                pos_link_y = -l_i * sp.cos(theta_i)
            else:
                pos_link_x = link_com_positions[i-1][0] + l_i * sp.sin(theta_i)
                pos_link_y = link_com_positions[i-1][1] - l_i * sp.cos(theta_i)
            
            link_com_positions.append((pos_link_x, pos_link_y))
            vel_com_squared = sp.diff(pos_link_x, t)**2 + sp.diff(pos_link_y, t)**2

            T_link = 0.5 * m_i * vel_com_squared
            V_link = m_i * g * pos_link_y

            link_kinetic_energies.append(T_link)
            link_potential_energies.append(V_link)

        T_total = T_cart + sum(link_kinetic_energies)
        V_total = sum(link_potential_energies)

        L = T_total - V_total
        # Equations of motion using Euler-Lagrange equations
        # d/dt(dL/dq_dot) - dL/dq - Q_i = 0
        equations = []
        motion_eq_cart_pos = sp.diff(sp.diff(L, sp.diff(cart_pos, t)), t) - sp.diff(L, cart_pos) - F_a
        equations.append(motion_eq_cart_pos)
        for i in range(self.num_links):
            theta_i = link_angles[i]
            motion_eq_link = sp.diff(sp.diff(L, sp.diff(theta_i, t)), t) - sp.diff(L, theta_i)
            equations.append(motion_eq_link)
        
        # simplify equations
        equations = [sp.simplify(eq) for eq in equations]

        # Define second derivatives as symbols (not functions!) to solve for them to get a non-linear state space
        xdd = sp.Symbol('xdd', real=True)
        theta_dd = [sp.Symbol(f'theta{i+1}_dd', real=True) for i in range(self.num_links)]

        # Replace second derivatives in equations
        xdd_expr = sp.diff(cart_pos, t, t)
        theta_dd_expr = [sp.diff(theta, t, t) for theta in link_angles]
        equations_with_double_derivatives = [eq.subs({xdd_expr: xdd, **{theta_dd_expr[i]: theta_dd[i] for i in range(self.num_links)}}) for eq in equations]
        equations_with_double_derivatives = [sp.simplify(eq) for eq in equations_with_double_derivatives]

        # Solve for second derivatives
        sol = sp.solve(equations_with_double_derivatives, (xdd, *theta_dd), simplify=True, rational=False)
        # Store the equations of motion
        self.equations_of_motion = sol

    def print_equations_of_motion(self):
        for var, eq in self.equations_of_motion.items():
            print(f"Equation for {var}:")
            sp.pprint(eq, use_unicode=True)


if __name__ == "__main__":
    num_links = 1
    link_lengths = [3.5]  # Length of each link
    link_masses = [2.0]  # Mass of each link
    cart_mass = 10.0
    pendulum = MultiLinkPendulum(num_links, link_lengths, link_masses, cart_mass)
    pendulum.print_equations_of_motion()