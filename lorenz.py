# A Manim implementation of the Lorenz attractor
# Author @yashpincha

from manim import *
import numpy as np
import os 

class LorenzAttractor(ThreeDScene):
    def construct(self):
        self.create_lorenz_attractor()

    def lorenz_system(self, pos, sigma=10, rho=28, beta=8/3):
        x, y, z = pos
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return np.array([dx_dt, dy_dt, dz_dt])

    def rate_to_color(self, rate, min_rate, max_rate):
        epsilon = 1e-6 
        rate_log = np.log(rate + epsilon)
        min_rate_log = np.log(min_rate + epsilon)
        max_rate_log = np.log(max_rate + epsilon)
        rate_normalized = (rate_log - min_rate_log) / (max_rate_log - min_rate_log)
        return interpolate_color(BLUE, PURPLE, rate_normalized)

    def create_lorenz_attractor(self):
        # Initial position
        pos = np.array([-1.0, 1.0, 0.0])

        # Parameters for the animation
        dt = 0.001
        steps = 40000
        scale_factor = 0.1
        dt_scaling_factor = 0.4

        # Create a curve from the initial position
        curve = ParametricFunction(
            lambda t: pos,
            t_range=[0, 1, 0.1],
            color=YELLOW,
        )

        min_rate = float("inf")
        max_rate = float("-inf")

        # Calculate min_rate and max_rate
        for _ in range(steps):
            dp = self.lorenz_system(pos) * dt
            rate = np.linalg.norm(dp)
            min_rate = min(min_rate, rate)
            max_rate = max(max_rate, rate)
            pos += dp

        # Reset initial position
        pos = np.array([-1.0, 1.0, 0.0])

        # Create segments with color based on rate of change
        for _ in range(steps):
            dp = self.lorenz_system(pos) * dt
            rate = np.linalg.norm(dp)
            segment_color = self.rate_to_color(rate, min_rate, max_rate)
            segment = Line(pos - dp, pos, color=segment_color, stroke_width=2)
            curve.add(segment)
            pos += dp

            # Adjust time step based on rate of change
            adaptive_dt = dt * dt_scaling_factor / rate
            pos += dp * adaptive_dt

        curve.scale(scale_factor)
        curve.move_to(ORIGIN)  # Center the curve in the scene

        # Add camera rotation
        self.set_camera_orientation(phi=80 * DEGREES, theta= 45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.8)
        lorenz_equation = MathTex(r"\begin{cases} \dot{x} = \sigma(y - x) \\ \dot{y} = x(\rho - z) - y \\ \dot{z} = xy - \beta z \end{cases}").scale(0.8)
        lorenz_equation.to_edge(UL)
        self.add_fixed_in_frame_mobjects(lorenz_equation)
        
        self.play(Write(lorenz_equation), run_time = 2)
        self.play(Create(curve), run_time=20, rate_func=linear)
        self.wait()

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if "--play" in sys.argv:
        script_name = f"{Path(__file__).resolve()}"
        os.system(f"manim -p -ql -i {script_name}")
