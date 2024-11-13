import numpy as np

class Plane:
    def __init__(self, normal, d):
        self.normal = np.array(normal, dtype=np.float64)
        self.d = d
        self.color=np.array([1.0,1.0,1.0])

    def intersect(self, ray_origin, ray_direction):
        # Calculate the denominator of the intersection formula
        denom = np.dot(self.normal, ray_direction)
        if abs(denom) < 1e-8:
            return None  # The ray is parallel to the plane and does not intersect

        # Calculate the intersection distance t
        t = -(np.dot(self.normal, ray_origin) + self.d) / denom
        if t >= 1e-8:  # Intersection is only valid if t is positive
            return t
        return None