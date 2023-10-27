# A Pillow implementation of the Lorenz attractor
# author @yashpincha

import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as IPdisplay
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob

# Set up the save folder for images
save_folder = 'images/lorenz-animate'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Define the initial system state (x, y, z positions in space)
initial_state = [0, 0, 0]

# Define the system parameters: sigma, rho, and beta
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Define the time points to solve for
start_time = 1
end_time = 60
interval = 200
time_points = np.linspace(start_time, end_time, end_time * interval)

# Define the Lorenz system
def lorenz_system(current_state, t):
    x, y, z = current_state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Plot the system in 3 dimensions
def plot_lorenz(coordinates, frame_number, save_folder):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    ax.plot(x, y, z, color='purple', alpha=0.7, linewidth=0.7)
    ax.set_xlim((-30, 30))
    ax.set_ylim((-30, 30))
    ax.set_zlim((0, 50))
    filename = f'{save_folder}/{frame_number:03d}.png'
    plt.savefig(filename, dpi=60, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Helper function to return a list in iteratively larger chunks
def get_chunks(full_list, chunk_size):
    chunk_size = max(1, chunk_size)
    chunks = [full_list[:i] for i in range(1, len(full_list) + 1, chunk_size)]
    return chunks

# Get incrementally larger chunks of the time points to reveal the attractor one frame at a time
chunk_size = 20
chunks = get_chunks(time_points, chunk_size)

# Get the points to plot, one chunk of time steps at a time, by integrating the system of equations
points = [odeint(lorenz_system, initial_state, chunk) for chunk in chunks]

# Plot each set of points, one at a time, saving each plot
for frame_number, point in enumerate(points):
    plot_lorenz(point, frame_number, save_folder)

# Create a tuple of display durations, one for each frame
first_last_duration = 100  # Show the first and last frames for 100 ms
standard_duration = 5  # Show all other frames for 5 ms
durations = (first_last_duration,) + (standard_duration,) * (len(points) - 2) + (first_last_duration,)

# Load all the static images into a list
image_files = glob.glob('{}/*.png'.format(save_folder))
images = [Image.open(image_file) for image_file in image_files]

# Set up the GIF file path
gif_filepath = 'images/animated-lorenz-attractor.gif'

# Save the images as an animated GIF
gif = images[0]
gif.info['duration'] = durations  # Duration in ms per frame
gif.info['loop'] = 0  # Loop infinitely
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])

# Verify that the number of frames in the GIF equals the number of image files and durations
gif_frames = Image.open(gif_filepath).n_frames
images_count = len(images)
durations_count = len(durations)

assert gif_frames == images_count == durations_count

# Display the animated GIF
IPdisplay.Image(url=gif_filepath)
