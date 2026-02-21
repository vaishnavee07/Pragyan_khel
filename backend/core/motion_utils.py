īimport math
import time

class CubicBezier:
    """
    Cubic Bezier easing function (CSS-style).
    Ported from standard web implementations.
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.cx = 3.0 * x1
        self.bx = 3.0 * (x2 - x1) - self.cx
        self.ax = 1.0 - self.cx - self.bx
        self.cy = 3.0 * y1
        self.by = 3.0 * (y2 - y1) - self.cy
        self.ay = 1.0 - self.cy - self.by

    def sample_curve_x(self, t):
        return ((self.ax * t + self.bx) * t + self.cx) * t

    def sample_curve_y(self, t):
        return ((self.ay * t + self.by) * t + self.cy) * t

    def sample_derivative_x(self, t):
        return (3.0 * self.ax * t + 2.0 * self.bx) * t + self.cx

    def solve(self, x, epsilon=1e-5):
        return self.sample_curve_y(self.solve_curve_x(x, epsilon))

    def solve_curve_x(self, x, epsilon):
        t0, t1, t2, x2, d2 = 0.0, 0.0, 0.0, 0.0, 0.0
        # First try a few iterations of Newton's method
        t2 = x
        for i in range(8):
            x2 = self.sample_curve_x(t2) - x
            if abs(x2) < epsilon:
                return t2
            d2 = self.sample_derivative_x(t2)
            if abs(d2) < 1e-6:
                break
            t2 = t2 - x2 / d2
        
        # Fallback to bi-section
        t0, t1 = 0.0, 1.0
        t2 = x
        if t2 < t0: return t0
        if t2 > t1: return t1
        
        while t0 < t1:
            x2 = self.sample_curve_x(t2)
            if abs(x2 - x) < epsilon:
                return t2
            if x > x2:
                t0 = t2
            else:
                t1 = t2
            t2 = (t1 - t0) * 0.5 + t0
        return t2

class SpringPhysics:
    """
    Spring simulation for "Apple-like" fluid motion.
    Based on damped harmonic oscillator.
    """
    def __init__(self, stiffness=170, damping=26, mass=1):
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass
        self.velocity = 0
        self.value = 0
        self.target = 0
        self.last_time = time.time()

    def update(self, target_val, dt=None):
        """
        Update the spring towards target_val.
        Returns the new position.
        """
        now = time.time()
        if dt is None:
            dt = now - self.last_time
        self.last_time = now
        
        # Cap dt to avoid instability
        dt = min(dt, 0.1)

        force = -self.stiffness * (self.value - target_val)
        damper = -self.damping * self.velocity
        acceleration = (force + damper) / self.mass

        self.velocity += acceleration * dt
        self.value += self.velocity * dt

        return self.value
    
    def reset(self, val):
        self.value = val
        self.velocity = 0
        self.target = val

# Pre-defined curves
EASE_OUT_QUART = CubicBezier(0.165, 0.84, 0.44, 1)
EASE_APPLE_IOS = CubicBezier(0.25, 0.1, 0.25, 1) # Generalized
EASE_CINEMATIC = CubicBezier(0.22, 1, 0.36, 1)   # Recommended in prompt
