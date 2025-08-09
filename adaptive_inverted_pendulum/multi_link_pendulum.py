import numpy as np
import sympy as sp

class MultiLinkPendulum:
    def __init__(self, num_links, link_lengths, link_masses, cart_mass, damping_coeff):
        self.num_links = num_links
        self.cart_mass = cart_mass
        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.g_val = 9.81  # gravitational acceleration in m/s^2
        self.damping_coeff = damping_coeff

        assert len(link_lengths) == num_links, "Number of link lengths must match num_links"
        assert len(link_masses) == num_links, "Number of link masses must match num_links"
        assert num_links > 0, "Number of links must be greater than 0"
        assert cart_mass > 0, "Cart mass must be greater than 0"

        # Build symbolic dynamics and compile fast numeric functions
        self.derive_non_linear_state_space()

        # state = [x, xdot, theta_1..theta_n, thetadot_1..thetadot_n]
        self.state = np.zeros(2 + 2 * num_links, dtype=float)

    def derive_non_linear_state_space(self):
        # Symbols and generalized coordinates
        t = sp.Symbol('t', real=True)
        g = sp.Symbol('g', real=True)              # gravitational acceleration
        M_cart = sp.Symbol('M_cart', real=True)    # cart mass
        F_a = sp.Symbol('F_a', real=True)          # applied horizontal force on cart
        b = sp.Symbol('b', real=True)              # cart viscous damping coeff

        cart_pos = sp.Function('x')(t)

        # Kinetic and potential energies
        T_cart = 0.5 * M_cart * sp.diff(cart_pos, t)**2

        link_angles = [sp.Function(f'theta_{i+1}')(t) for i in range(self.num_links)]
        link_com_positions = []
        link_kinetic_energies = []
        link_potential_energies = []

        # per-link length/mass symbols
        l_syms = [sp.Symbol(f'l_{i+1}', real=True) for i in range(self.num_links)]
        m_syms = [sp.Symbol(f'm_{i+1}', real=True) for i in range(self.num_links)]

        for i in range(self.num_links):
            l_i = l_syms[i]
            m_i = m_syms[i]
            theta_i = link_angles[i]

            if i == 0:
                pos_link_x = cart_pos + l_i * sp.sin(theta_i)
                pos_link_y = -l_i * sp.cos(theta_i)
            else:
                pos_link_x = link_com_positions[i - 1][0] + l_i * sp.sin(theta_i)
                pos_link_y = link_com_positions[i - 1][1] - l_i * sp.cos(theta_i)

            link_com_positions.append((pos_link_x, pos_link_y))

            vx = sp.diff(pos_link_x, t)
            vy = sp.diff(pos_link_y, t)
            vel_com_squared = vx**2 + vy**2

            T_link = 0.5 * m_i * vel_com_squared
            V_link = m_i * g * pos_link_y

            link_kinetic_energies.append(T_link)
            link_potential_energies.append(V_link)

        T_total = T_cart + sum(link_kinetic_energies)
        V_total = sum(link_potential_energies)
        L = T_total - V_total

        # Euler-Lagrange equations (cart has external force and viscous damping bx_dot)
        equations = []
        motion_eq_cart_pos = sp.diff(sp.diff(L, sp.diff(cart_pos, t)), t) - sp.diff(L, cart_pos) - F_a + b * sp.diff(cart_pos, t)
        equations.append(sp.simplify(motion_eq_cart_pos))

        for i in range(self.num_links):
            theta_i = link_angles[i]
            motion_eq_link = sp.diff(sp.diff(L, sp.diff(theta_i, t)), t) - sp.diff(L, theta_i)
            equations.append(sp.simplify(motion_eq_link))

        # Introduce algebraic symbols for second derivatives
        xdd = sp.Symbol('xdd', real=True)
        theta_dd = [sp.Symbol(f'theta{i+1}_dd', real=True) for i in range(self.num_links)]

        # Replace second derivatives to form algebraic system
        xdd_expr = sp.diff(cart_pos, t, t)
        theta_dd_exprs = [sp.diff(theta, t, t) for theta in link_angles]

        eqs_algebraic = []
        for eq in equations:
            eq_sub = eq.subs({xdd_expr: xdd})
            for j in range(self.num_links):
                eq_sub = eq_sub.subs({theta_dd_exprs[j]: theta_dd[j]})
            eqs_algebraic.append(sp.simplify(eq_sub))

        # Solve for accelerations
        sol = sp.solve(eqs_algebraic, (xdd, *theta_dd), simplify=True, rational=False)

        # Build algebraic (non-functional) symbols for current value substitution
        self.x_sym = sp.Symbol('x', real=True)
        self.xdot_sym = sp.Symbol('xdot', real=True)
        self.theta_syms = [sp.Symbol(f'theta_{i+1}', real=True) for i in range(self.num_links)]
        self.thetadot_syms = [sp.Symbol(f'thetadot_{i+1}', real=True) for i in range(self.num_links)]
        self.F_a_sym = F_a
        self.g_sym = g
        self.M_cart_sym = M_cart
        self.l_syms = l_syms
        self.m_syms = m_syms
        self.b = b

        # Map functional vars -> algebraic symbols (positions & velocities)
        subs_map = {
            cart_pos: self.x_sym,
            sp.diff(cart_pos, t): self.xdot_sym,
        }
        for i in range(self.num_links):
            subs_map[link_angles[i]] = self.theta_syms[i]
            subs_map[sp.diff(link_angles[i], t)] = self.thetadot_syms[i]

        # Expressions for accelerations using algebraic symbols
        xdd_algebraic = sp.simplify(sol[xdd].subs(subs_map))
        theta_dds_algebraic = [sp.simplify(sol[theta_dd[i]].subs(subs_map)) for i in range(self.num_links)]

        # --------- SPEED-UP: compile one NumPy function for all accelerations ----------
        # Order of inputs to the numeric function:
        # [x, xdot, theta_1..theta_n, thetadot_1..thetadot_n, F_a, g, M_cart, b, l_1..l_n, m_1..m_n]
        self._state_syms_order = [self.x_sym, self.xdot_sym] + self.theta_syms + self.thetadot_syms
        self._param_syms_order = [self.F_a_sym, self.g_sym, self.M_cart_sym, self.b] + self.l_syms + self.m_syms
        vars_order = self._state_syms_order + self._param_syms_order

        accel_vec = sp.Matrix([xdd_algebraic] + theta_dds_algebraic)
        # Use 'numpy' backend; keep it simple (no numba dependency)
        self._accel_func = sp.lambdify(vars_order, accel_vec, 'numpy')

        # Keep expressions if you still want to inspect/print them externally
        self.xdd_expr = xdd_algebraic
        self.theta_dd_exprs = theta_dds_algebraic
        self.equations_of_motion = {xdd: xdd_algebraic}
        for i, thdd in enumerate(theta_dds_algebraic):
            self.equations_of_motion[sp.Symbol(f'theta{i+1}_dd')] = thdd

    def step_in_time(self, F_a_val: float, dt: float):
        """
        Advance one step using a simple semi-implicit (symplectic) Euler integrator.
        State layout: [x, xdot, theta_1..theta_n, thetadot_1..thetadot_n]
        """
        # Unpack
        x = float(self.state[0])
        xdot = float(self.state[1])
        thetas = self.state[2:2 + self.num_links].astype(float, copy=True)
        thetadots = self.state[2 + self.num_links:].astype(float, copy=True)

        # Build ordered numeric argument vector once (strictly follow vars_order)
        args = []
        # state
        args.extend([x, xdot])
        args.extend(thetas.tolist())
        args.extend(thetadots.tolist())
        # params
        args.append(float(F_a_val))
        args.append(float(self.g_val))
        args.append(float(self.cart_mass))
        args.append(float(self.damping_coeff))
        args.extend([float(L) for L in self.link_lengths])
        args.extend([float(m) for m in self.link_masses])

        # Fast numeric evaluation
        accels = np.asarray(self._accel_func(*args), dtype=float).reshape(-1)
        xdd_num = accels[0]
        theta_dds_num = accels[1:]

        # Semi-implicit Euler: update velocities, then positions
        xdot += xdd_num * dt
        thetadots += theta_dds_num * dt
        x += xdot * dt
        thetas += thetadots * dt

        # write back
        self.state[0] = x
        self.state[1] = xdot
        self.state[2:2 + self.num_links] = thetas
        self.state[2 + self.num_links:] = thetadots

        return self.state.copy()


if __name__ == "__main__":
    # Example: 2-link system (same external interface as your original)
    num_links = 2
    link_lengths = [3.5] * num_links
    link_masses = [2.0]  * num_links
    cart_mass = 10.0
    damping_coeff = 0.5

    pendulum = MultiLinkPendulum(num_links, link_lengths, link_masses, cart_mass, damping_coeff)

    # step the pendulum
    dt = 0.01  # time step in seconds
    force = 50
    for _ in range(100):  # simulate 1 second
        state = pendulum.step_in_time(force, dt)
        print(f"State: {state}")
    print("Simulation complete.")
