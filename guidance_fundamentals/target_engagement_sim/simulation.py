import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from guidance_fundamentals.target_engagement_sim.vector3D import Vector3D
from guidance_fundamentals.target_engagement_sim.point_mass_model import PointMassModel3D
from guidance_fundamentals.target_engagement_sim.controller import DummyController


class TargetEngagementSimulation:
    def __init__(
        self,
        pursuer_initial_pos: Vector3D,
        pursuer_initial_vel: Vector3D,
        pursuer_max_accel: np.ndarray,
        pursuer_delay_tau: np.ndarray,
        target_initial_pos: Vector3D,
        target_initial_vel: Vector3D,
        target_max_accel: np.ndarray,
        target_delay_tau: np.ndarray,
        dt: float,
        total_time: float,
    ):
        # Initialize pursuer and target models
        self.pursuer = PointMassModel3D(
            pursuer_initial_pos, pursuer_initial_vel, Vector3D.from_array(pursuer_max_accel), pursuer_delay_tau
        )
        self.target = PointMassModel3D(
            target_initial_pos, target_initial_vel, Vector3D.from_array(target_max_accel), target_delay_tau
        )
        
        # Initialize controllers
        self.pursuer_controller = DummyController()
        self.target_controller = DummyController()
        
        # Simulation parameters
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        
        # History storage
        self.pursuer_history = []
        self.target_history = []
        self.time_history = []
        
    def run(self):
        """Run the simulation and store trajectory history."""
        current_time = 0.0
        
        for step in range(self.num_steps):
            # Get current states
            pursuer_pos, pursuer_vel = self.pursuer.get_state()
            target_pos, target_vel = self.target.get_state()
            
            # Store history
            self.pursuer_history.append((pursuer_pos.x, pursuer_pos.y, pursuer_pos.z))
            self.target_history.append((target_pos.x, target_pos.y, target_pos.z))
            self.time_history.append(current_time)
            
            # Compute acceleration commands
            pursuer_accel_cmd = self.pursuer_controller.compute_acceleration_command(
                (pursuer_pos, pursuer_vel), (target_pos, target_vel), current_time
            )
            target_accel_cmd = self.target_controller.compute_acceleration_command(
                (target_pos, target_vel), (pursuer_pos, pursuer_vel), current_time
            )
            
            # Step both models
            self.pursuer.step(pursuer_accel_cmd, self.dt)
            self.target.step(target_accel_cmd, self.dt)
            
            # Update time
            current_time += self.dt
            
        # Convert history to numpy arrays for easier plotting
        self.pursuer_history = np.array(self.pursuer_history)
        self.target_history = np.array(self.target_history)
        self.time_history = np.array(self.time_history)
        
    def animate(self, interval=50, trail_length=50):
        """Create an animated 3D visualization of the engagement."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        pursuer_point, = ax.plot([], [], [], 'ro', markersize=12, label='Pursuer')
        target_point, = ax.plot([], [], [], 'bs', markersize=12, label='Target')
        pursuer_trail, = ax.plot([], [], [], 'r-', alpha=0.6, linewidth=2)
        target_trail, = ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2)
        
        # Set axis labels
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Target Engagement Simulation')
        ax.legend()
        
        # Set axis limits based on trajectory data
        all_positions = np.vstack([self.pursuer_history, self.target_history])
        
        # Calculate margin as 10% of the range or 500m, whichever is larger
        x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
        y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
        z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
        
        x_margin = max(500, x_range * 0.1)
        y_margin = max(500, y_range * 0.1)
        z_margin = max(500, z_range * 0.1)
        
        ax.set_xlim([all_positions[:, 0].min() - x_margin, all_positions[:, 0].max() + x_margin])
        ax.set_ylim([all_positions[:, 1].min() - y_margin, all_positions[:, 1].max() + y_margin])
        ax.set_zlim([all_positions[:, 2].min() - z_margin, all_positions[:, 2].max() + z_margin])
        
        # Add time text
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        distance_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
        
        def init():
            pursuer_point.set_data([], [])
            pursuer_point.set_3d_properties([])
            target_point.set_data([], [])
            target_point.set_3d_properties([])
            pursuer_trail.set_data([], [])
            pursuer_trail.set_3d_properties([])
            target_trail.set_data([], [])
            target_trail.set_3d_properties([])
            time_text.set_text('')
            distance_text.set_text('')
            return pursuer_point, target_point, pursuer_trail, target_trail, time_text, distance_text
        
        def update(frame):
            # Update current positions
            pursuer_point.set_data([self.pursuer_history[frame, 0]], [self.pursuer_history[frame, 1]])
            pursuer_point.set_3d_properties([self.pursuer_history[frame, 2]])
            
            target_point.set_data([self.target_history[frame, 0]], [self.target_history[frame, 1]])
            target_point.set_3d_properties([self.target_history[frame, 2]])
            
            # Update trails (show last trail_length points)
            trail_start = max(0, frame - trail_length)
            pursuer_trail.set_data(
                self.pursuer_history[trail_start:frame+1, 0],
                self.pursuer_history[trail_start:frame+1, 1]
            )
            pursuer_trail.set_3d_properties(self.pursuer_history[trail_start:frame+1, 2])
            
            target_trail.set_data(
                self.target_history[trail_start:frame+1, 0],
                self.target_history[trail_start:frame+1, 1]
            )
            target_trail.set_3d_properties(self.target_history[trail_start:frame+1, 2])
            
            # Update time text
            time_text.set_text(f'Time: {self.time_history[frame]:.2f} s')
            
            # Calculate and display distance
            dist = np.linalg.norm(self.pursuer_history[frame] - self.target_history[frame])
            distance_text.set_text(f'Distance: {dist:.2f} m')
            
            return pursuer_point, target_point, pursuer_trail, target_trail, time_text, distance_text
        
        # Create animation
        anim = FuncAnimation(
            fig, update, init_func=init,
            frames=len(self.time_history),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        plt.show()
        
        return anim


def main():
    """Example simulation run."""
    # Initial conditions
    pursuer_pos = Vector3D(0.0, 0.0, 0.0)
    pursuer_vel = Vector3D(100.0, 50.0, 20.0)  # m/s
    
    target_pos = Vector3D(5000.0, 3000.0, 1000.0)  # m
    target_vel = Vector3D(-50.0, -30.0, 10.0)  # m/s
    
    # Physical parameters
    pursuer_max_accel = np.array([30.0, 30.0, 30.0])  # m/s^2
    pursuer_delay_tau = np.array([0.1, 0.1, 0.1])  # s
    
    target_max_accel = np.array([20.0, 20.0, 20.0])  # m/s^2
    target_delay_tau = np.array([0.1, 0.1, 0.1])  # s
    
    # Simulation parameters
    dt = 0.1  # s
    total_time = 60.0  # s
    
    # Create and run simulation
    sim = TargetEngagementSimulation(
        pursuer_pos, pursuer_vel, pursuer_max_accel, pursuer_delay_tau,
        target_pos, target_vel, target_max_accel, target_delay_tau,
        dt, total_time
    )
    
    print("Running simulation...")
    sim.run()
    print(f"Simulation complete. Simulated {len(sim.time_history)} time steps.")
    
    # Debug: print first and last positions
    print(f"\nPursuer trajectory:")
    print(f"  Start: {sim.pursuer_history[0]}")
    print(f"  End: {sim.pursuer_history[-1]}")
    print(f"\nTarget trajectory:")
    print(f"  Start: {sim.target_history[0]}")
    print(f"  End: {sim.target_history[-1]}")
    
    # Calculate final separation correctly (for single points)
    final_separation = np.linalg.norm(sim.pursuer_history[-1] - sim.target_history[-1])
    print(f"Final separation: {final_separation:.2f} m")
    
    # Create animation
    print("Creating animation...")
    sim.animate(interval=20, trail_length=100)


if __name__ == "__main__":
    main()
