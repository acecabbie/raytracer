import math

import numpy as np
from PIL import Image

# Define a Sphere class
class Sphere:
    def __init__(self, center, radius, color,shininess,texture_filename="none",pixelTestx=0,pixelTesty=0):

        self.center = np.array(center, dtype=np.float64)
        self.radius = np.array(radius,dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.shininess = shininess
        self.pixelTestx=pixelTestx
        self.pixelTesty=pixelTesty
        self.texture_filename = texture_filename
        self.texture_image = None

        if texture_filename and texture_filename.lower() != "none":
            self.load_texture(texture_filename)

    def load_texture(self, texture_filename):
        """Load the texture image using PIL and store it."""
        try:
            self.texture_image = Image.open(texture_filename).convert("RGB")
        except Exception as e:
            print(f"Failed to load texture {texture_filename}: {e}")
            self.texture_image = None

    def get_texture_color(self, intersection_point,g,h):
        """Get the color from the texture map based on the intersection point."""
        if not self.texture_image:
            return np.array(self.color)  # Default to the base color if no texture

        # Calculate latitude and longitude of the intersection point
        direction = intersection_point - self.center
        direction = direction / np.linalg.norm(direction)  # Normalize

        # Latitude and longitude calculations
        texture_width, texture_height = self.texture_image.size
        u = 0.5 - (np.arctan2(direction[2], direction[0]) / (2 * np.pi))
        #u = 0.5 + (math.atan2(direction[2], direction[0]) / (2 * np.pi))
       # u = (intersection_point[0] / texture_width) * 2 * math.pi - math.pi
        v = 0.5 - (np.arcsin(direction[1]) / np.pi)
        #latitude = (y / height) * math.pi - (math.pi / 2)

        # Convert (u, v) to texture image coordinates
        texture_width, texture_height = self.texture_image.size
        tex_x = int(u * texture_width) % texture_width
        tex_y = int(v * texture_height) % texture_height
        if (g == self.pixelTestx and h == self.pixelTesty):
            print("pixel x: ", tex_x, " pixel y: ", tex_y)
            print("pixel u: ", u, " pixel v: ", v)
            print("x^: ", tex_x * texture_width, tex_y * texture_height)
            print("direction: ",direction)
            print("intersection point: ",intersection_point)
        # Get the texture color
        texture_color = np.array(self.texture_image.getpixel((tex_x, tex_y))) / 255.0
        color= self.srgb_to_linear(texture_color)
        return color




    def srgb_to_linear(self,color):
        """Convert sRGB color to linear RGB."""
        color = np.clip(color, 0.0, 1.0)
        return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
    def get_texture_colortwo(self, P,g,h):

        # Calculate spherical coordinates
        x, y, z = P - self.center
        theta = math.atan2(z, x)
        phi = math.acos(y / self.radius)

        # Convert to texture coordinates (u, v)
        u = (theta + math.pi) / (2 * math.pi)
        v = phi / math.pi

        # Map (u, v) to pixel coordinates
        img_width, img_height = self.texture_image.size
        pixel_x = int(u * img_width) % img_width
        pixel_y = int((1 - v) * img_height) % img_height  # Flip v to match image coordinates

        if (g == self.pixelTestx and h == self.pixelTesty):
            print("pixel x: ",x," pixel y: ",y)
            print("pixel u: ", u, " pixel v: ", v)
            print("x^: ",x*img_width,y*img_height)
        # Get the color from the texture
        color = self.texture_image.getpixel((pixel_x, pixel_y))
        return np.array(color) / 255.0  # Normalize to [0, 1]
    def get_texture_colorthree(self,P):
        print("s")
    def intersect(self, ray_origin, ray_dir,x,y):
        ray_origin=np.array(ray_origin,dtype=np.float64)

        ray_dir = np.array(ray_dir, dtype=np.float64)
        self.center = np.array(self.center, dtype=np.float64)
        # Step 1: Calculate the vector from the ray origin to the sphere center
        oc =  self.center - ray_origin
        # Calculate the squared radius for easy comparison later
        radius2 = self.radius ** 2

        # Step 1: Check if the ray origin is inside the sphere
        inside = np.dot(oc, oc) < radius2

        # Step 2: Calculate tc, the distance along the ray to the closest approach to the sphere's center
        #tc = np.dot(oc, ray_dir)  # tc is the projection of oc onto ray_dir
        tc = np.dot(oc, ray_dir) / np.linalg.norm(ray_dir)
        #if (x == self.pixelTestx and y == self.pixelTesty):
         #   print("closet point: ",np.round(np.dot(oc, oc),5), radius2, inside, tc)
        # Step 3: If outside and tc is negative, there's no intersection
        if not inside and tc < 0:


            return None

        # Step 4: Calculate d^2, the squared distance from the center of the sphere to the point of closest approach
        #d2 = np.dot(oc, oc) - tc ** 2
        closest_point = ray_origin + tc * ray_dir

        co = closest_point - self.center

        d2 = np.round(np.dot(co, co),4)
        # Step 5: If outside and d^2 is greater than r^2, no intersection
        if not inside and radius2 < d2:

            return None

        # Step 6: Calculate toffset
        toffset = np.sqrt(radius2 - d2)/np.linalg.norm(ray_dir)

        # Step 7: Determine intersection distances
        if inside:
            # Intersection point is exiting the sphere
            t = tc + toffset
        else:
            # Two intersection points; we take the closer one (entering the sphere)
            t = tc - toffset #if tc - toffset > 0 else tc + toffset

        return t if t > 0 else None  # Return None if intersection is behind the ray origin
    def intersect_t(self, ray_origin, ray_dir):
        """Ray-sphere intersection function.
        Returns the distance from the ray origin to the intersection point or None if no intersection."""
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None  # No intersection
        else:
            return (-b - np.sqrt(discriminant)) / (2.0 * a)