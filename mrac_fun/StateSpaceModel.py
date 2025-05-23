import numpy as np

class StateSpaceModel:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.state = np.zeros(A.shape[0]).reshape(-1, 1)  # Initialize state vector to zeros  

    def update(self, u, dt):
        x_dot = self.A @ self.state + self.B @ u
        self.state = x_dot * dt + self.state

    def output(self):
        return self.C @ self.state
    
    def output_with_u(self, u):
        return self.C @ self.state + self.D @ u