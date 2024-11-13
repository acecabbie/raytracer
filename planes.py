
import math

from GPUSettings import GPUSettings,Ray
from Sphere import Sphere
from plane import Plane
import numpy as np
from PIL import Image


pixelTestx=7
pixelTesty=42

gpusettings=GPUSettings()


def save_image(image, width, height, filename):
    #img = Image.fromarray(image, 'RGBA')
    #img = img.resize((width, height), Image.NEAREST)
    print("saving image: ",filename)
    image.save(filename)


def trace_ray(ray_origin, ray_dir, spheres, planes, x, y, depth=1, max_bounces=10):
    if depth > max_bounces:
        return np.array([0, 0, 0])  # Return black if the recursion depth is exceeded

    background_color = np.array([0, 0, 0, 0])
    color = background_color
    nearest_t = float('inf')
    hit_object = None
    epsilon = 1e-5
    P = None
    normal = None

    # Check for intersections with spheres
    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_dir, x, y)
        if t and t < nearest_t:
            nearest_t = t
            hit_object = sphere
            P = ray_origin + t * ray_dir
            normal = (P - sphere.center) / np.linalg.norm(P - sphere.center)
            if np.dot(ray_dir, normal) > 0:
                normal = -normal

    # Check for intersections with planes
    for plane in planes:
        t = plane.intersect(ray_origin, ray_dir)
        if t and t < nearest_t:
            nearest_t = t
            hit_object = plane
            P = ray_origin + t * ray_dir
            normal = plane.normal/np.linalg.norm(plane.normal)
            if np.dot(ray_dir, normal) > 0:
                normal = -normal

    if hit_object is None:
        return background_color  # No intersection, return background color

    final_color = np.array([0.0, 0.0, 0.0])

    # Lighting and shadows
    for light_direction, light_color in zip(gpusettings.light_directions, gpusettings.light_colors):
        in_shadow = False
        shadow_ray_origin = P + normal * epsilon

        # Check shadows for spheres
        for sphere in spheres:
            if sphere != hit_object:
                shadow_t = sphere.intersect(shadow_ray_origin, light_direction, x, y)
                if shadow_t and shadow_t >= epsilon:
                    in_shadow = True
                    break

        # Check shadows for planes
        for plane in planes:
            if plane != hit_object:
                shadow_t = plane.intersect(shadow_ray_origin, light_direction)
                if shadow_t and shadow_t >= epsilon:
                    in_shadow = True
                    break

        # If no object casts a shadow, calculate lighting
        if not in_shadow:
            diffuse_intensity = max(np.dot(normal, light_direction), 0)
            final_color += np.array(hit_object.color) * diffuse_intensity * np.array(light_color)
            if(x==pixelTestx and y==pixelTesty):
                print("color not in shadow: ",final_color,hit_object.color)
                print("color not in shadow: ", diffuse_intensity, normal,light_direction)
                print("color not in shadow: ", final_color, light_color)

    # Reflections
    if isinstance(hit_object, Sphere) and hit_object.shininess and max(hit_object.shininess) > 0 and depth < max_bounces:
        # Compute reflection only if we haven't reached max depth
        reflection_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
        reflection_dir /= np.linalg.norm(reflection_dir)  # Normalize
        reflection_color = trace_ray(P + normal * epsilon, reflection_dir, spheres, planes, x, y, depth + 1, max_bounces) / 255
        reflection_color = reflection_color[:3] if len(reflection_color) > 3 else reflection_color
        reflection_factor = np.array(hit_object.shininess)
        final_color = (1 - reflection_factor) * final_color + reflection_factor * reflection_color

    color = final_color * 255
    # Always apply linear to sRGB conversion for the final color at the top level
    if depth == 1:
        color = linear_to_srgb(final_color) * 255

    return np.clip(color, 0, 255).astype(np.uint8)






          #  light_dir = np.array([1.0, -1.0, -1.0])
           # light_dir = light_dir / np.linalg.norm(light_dir)
           # diffuse_intensity = max(np.dot(normal, light_dir), 0)
           # color = sphere.color * diffuse_intensity * 255  # Scale for RGB
    #return (np.clip(color, 0, 255)).astype(np.uint8)
def linear_to_srgb(linear_color):
    """Convert linear RGB color to sRGB"""
    srgb_color = np.where(
        linear_color <= 0.0031308,
        12.92 * linear_color,
        1.055 * np.power(linear_color, 1 / 2.4) - 0.055
    )
    return srgb_color

# Main rendering function
def render(spheres,planes, image_size, output_filename):
    print("suns: ",gpusettings.light_directions)
    print("sun colors: ",gpusettings.light_colors)

    width, height = image_size
    eye = np.array([0, 0, 0],dtype=np.float64)  # Camera position
    forward = np.array([0, 0, -1],dtype=np.float64)  # Forward direction
    right = np.array([1, 0, 0],dtype=np.float64)  # Right direction
    up = np.array([0, 1, 0],dtype=np.float64)  # Up direction

    image = Image.new("RGBA", (width, height))
    pixels = image.load()
    #ray=Ray()
    # Loop through each pixel
    for y in range(height):
        for x in range(width):
            # Compute sx and sy for each pixel
            sx = (2 * x - width) / max(width, height)
            sy = (height - 2 * y) / max(width, height)
            if(gpusettings.fish_eye):
                print("sx2 and sy2 and total: ",sx**2,sy**2,sx**2+sy**2)
                if sx ** 2 + sy ** 2 > 1:
                    continue  # Skip this pixel, as it’s outside the fisheye range
                print("forward factor: ", 1 - (sx ** 2) - (sy ** 2),x,y)
                forwardFactor = math.sqrt(max(0, 1 - (sx ** 2) - (sy ** 2)))          # Compute the direction of the ray based on sx and sy
                print("forward factor: ",forwardFactor)

                direction = sx * right + sy * up + forwardFactor * forward
            else:
                direction = forward + sx * right + sy * up
            if(gpusettings.panorama):


                # Map x and y to longitude and latitude
                longitude = (x / width) * 2 * math.pi - math.pi  # Range: [-π, π]
                latitude = (y / height) * math.pi - (math.pi / 2)  # Range: [-π/2, π/2]
                latitude = (1 - y / height) * math.pi - (math.pi / 2)  # Reverse the range for y

                # Compute the direction of the ray in spherical coordinates
                dir_x = math.cos(latitude) * math.sin(longitude)
                dir_y = math.sin(latitude)
                dir_z = math.cos(latitude) * math.cos(longitude)

                # Assemble the direction vector
                direction = dir_x * right + dir_y * up + dir_z * forward
            direction = direction / np.linalg.norm(direction)  # Normalize the direction

            # Set the ray origin to the camera's position
            ray_origin = eye
            ray_direction = direction


            color = trace_ray(ray_origin, ray_direction, spheres,planes,x,y)
            if (x == pixelTestx and y == pixelTesty):
                print("fin cilor: ", color)
            pixels[x, y] = tuple(color)  # Set the pixel color
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
# Basic function to read input file and set up scene
def read_scene(filename):
    index=1
    spheres = []
    planes = []
    image_size = (100, 50)  # Default
    color=None
    shininess=None
    output_filename = "output.png"  # Default
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "png":
                image_size = (int(parts[1]), int(parts[2]))
                print("image size: ",image_size)
                output_filename = parts[3]
            elif parts[0] == "bounces":
                gpusettings.bounces=int(parts[1])
                print("gpu yo: ",gpusettings.bounces)
            elif parts[0] == "color":
                color = list(map(float, parts[1:4]))
            elif parts[0] == "shininess":
                #print("length: ",len(parts))
                if len(parts) == 2:
                    # Single value given, use it for all color channels
                    shininess = [float(parts[1])] * 3
                    #print("red: ",shininess)
                else:
                    # Three values given
                    shininess = list(map(float, parts[1:4]))
                    #print("blue: ",shininess)


            elif parts[0] == "sphere":
                center = list(map(float, parts[1:4]))
                radius = float(parts[4])
                if(not color):
                    color = [1.0, 1.0, 1.0]  # Default color (red)
                print("shininess: ",shininess)
                spheres.append(Sphere(center, radius, color,shininess,pixelTestx,pixelTesty,index))
                index+=1
            elif parts[0] == "plane":
                normal = list(map(float, parts[1:4]))
                d = float(parts[4])
                planes.append(Plane(normal, d))

            elif parts[0] == "sun":
                sun_direction = np.array(list(map(float, parts[1:4])))
                if(not color):
                    color = [1.0, 1.0, 1.0]
                print("sun color: ",color)
                gpusettings.light_color=color
                #print("sun direction: ",sun_direction)
                sun_direction = sun_direction / np.linalg.norm(sun_direction)  # Normalize
                gpusettings.light_direction=sun_direction
                sun_direction = np.array(list(map(float, parts[1:4])))
                sun_direction = sun_direction / np.linalg.norm(sun_direction)  # Normalize
                gpusettings.light_directions.append(sun_direction)
                gpusettings.light_colors.append(color if color else [1.0, 1.0, 1.0])
            elif parts[0] == "fisheye":
                gpusettings.fish_eye=True
            elif parts[0] == "panorama":
                gpusettings.panorama=True
    return spheres,planes, image_size, output_filename
# Example usage
if __name__ == "__main__":
    spheres,planes, image_size, output_filename = read_scene("ray-shadow-plane.txth")

    render(spheres,planes, image_size, output_filename)