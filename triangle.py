import numpy as np
from PIL import Image

class Triangle:
    def __init__(self, v1, v2, v3, color,texture_filename,texcoords):
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.v3 = np.array(v3)
        self.color = np.array(color)

        # Calculate the edges
        self.edge1 = self.v2 - self.v1
        self.edge2 = self.v3 - self.v1

        # Compute the normal using the cross product
        self.normal = np.cross(self.edge1, self.edge2)
        # Normalize the normal vector
        self.normal = self.normal / np.linalg.norm(self.normal)
        if texture_filename and texture_filename.lower() != "none":
            self.load_texture(texture_filename)
        self.texcoords = texcoords if texcoords else [[0, 0], [0, 0], [0, 0]]
    def load_texture(self, texture_filename):
        """Load the texture image using PIL and store it."""
        try:
            self.texture_image = Image.open(texture_filename).convert("RGB")
        except Exception as e:
            print(f"Failed to load texture {texture_filename}: {e}")
            self.texture_image = None

    def intersect(self, ray_origin, ray_dir):
        # Use the Möller–Trumbore intersection algorithm
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)
        epsilon = 1e-8
        if -epsilon < a < epsilon:
            return None,None,None  # This ray is parallel to the triangle.

        f = 1.0 / a
        s = ray_origin - self.v1
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None,None,None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)
        if v < 0.0 or u + v > 1.0:
            return None,None,None

        t = f * np.dot(edge2, q)
        if t > epsilon:  # Ray intersection
            return t,u,v
        return None,None,None
    def get_texture_color(self, u, v):
        if not self.texture_image:
            return self.color  # Default to the base color if no texture

        # Use barycentric coordinates (u, v) to interpolate texture coordinates
        texcoord1 = np.array(self.texcoords[0])
        texcoord2 = np.array(self.texcoords[1])
        texcoord3 = np.array(self.texcoords[2])

        # Interpolate texture coordinates
        texcoord = (1 - u - v) * texcoord1 + u * texcoord2 + v * texcoord3
        tex_u = texcoord[0]
        tex_v = texcoord[1]

        # Map (tex_u, tex_v) to texture image coordinates
        texture_width, texture_height = self.texture_image.size
        tex_x = int(tex_u * texture_width) % texture_width
        tex_y = int((1 - tex_v) * texture_height) % texture_height  # Flip v


        # Get the texture color and normalize it
        texture_color = np.array(self.texture_image.getpixel((tex_x, tex_y))) / 255.0
        return texture_color

    def get_texture_colortwo(self,P):
        lambda1, lambda2, lambda3=self.barycentric(P)
        if not self.texture_image:
            return self.color  # Default to the base color if no texture

        # Interpolate texture coordinates using barycentric coordinates
        texcoord1 = np.array(self.texcoords[0])
        texcoord2 = np.array(self.texcoords[1])
        texcoord3 = np.array(self.texcoords[2])

        # Interpolated texture coordinates
        texcoord = lambda1 * texcoord1 + lambda2 * texcoord2 + lambda3 * texcoord3
        tex_u, tex_v = texcoord

        # Map (tex_u, tex_v) to texture image coordinates
        texture_width, texture_height = self.texture_image.size
        tex_x = int(tex_u * texture_width) % texture_width
        tex_y = int((1 - tex_v) * texture_height) % texture_height  # Flip v
        tex_y=int(tex_v * texture_height) % texture_height


        # Get the texture color and normalize it
        texture_color = np.array(self.texture_image.getpixel((tex_x, tex_y))) / 255.0
        return texture_color
    def barycentric(self,P):

        # Define the vertices and point
        A = self.v1
        B = self.v2
        C = self.v3


        # Edge vectors
        e1 = B - A
        e2 = C - A
        p = P - A

        # Dot products
        dot11 = np.dot(e1, e1)
        dot12 = np.dot(e1, e2)
        dot22 = np.dot(e2, e2)
        dot1p = np.dot(e1, p)
        dot2p = np.dot(e2, p)

        # Calculate barycentric coordinates
        denom = dot11 * dot22 - dot12 * dot12
        lambda2 = (dot22 * dot1p - dot12 * dot2p) / denom
        lambda3 = (dot11 * dot2p - dot12 * dot1p) / denom
        lambda1 = 1 - lambda2 - lambda3
        return lambda1,lambda2,lambda3
    def srgb_to_linear(self,color):
        """Convert sRGB color to linear RGB."""
        color = np.clip(color, 0.0, 1.0)
        return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)