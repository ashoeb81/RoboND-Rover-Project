import numpy as np
import cv2


# Identify pixels between lower and upper thresholds
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
# Threshold  100 < R < 255, 100 < G < 255, B < 10 is good for identifying rocks.
# Threshold R < 185, G < 185, B < 185 is good for identifying obstacles.
def color_thresh(img, rgb_thresh_low=(160, 160, 160), rgb_thresh_high=(255, 255, 255)):
    mask = cv2.inRange(img, rgb_thresh_low, rgb_thresh_high)
    return mask


# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Convert pixels in rover-centric coordinates to a binary image with pixels in camera coordinates.
def image_coords(xpos, ypos, image_shape):
    out_img = np.zeros(image_shape)
    x_pixel = -ypos + image_shape[0]
    y_pixel = image_shape[0] - xpos
    out_img[np.int_(y_pixel), np.int_(x_pixel)] = 255
    return out_img


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw / 180. * np.pi
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated


# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = xpos + xpix_rot / scale
    ypix_translated = ypos + ypix_rot / scale
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    return warped


# Define a function that returns pixels in the field of view (fov) of
# the rover camera.
def pixels_in_fov(xpos, ypos, fov=np.pi / 4):
    rho, theta = to_polar_coords(xpos, ypos)
    theta_in_fov = np.abs(theta) < fov
    dist_in_fov = (np.abs(rho) < 150) & (np.abs(rho) > 10)
    return xpos[theta_in_fov & dist_in_fov], ypos[theta_in_fov & dist_in_fov]


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    dst_size = 5
    bottom_offset = 6
    src = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32([[Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1] / 2 + dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                      [Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                      ])
    warped = perspect_transform(Rover.img, src, dst)

    #1) Increment visit count for current location of the rover
    rover_y, rover_x = [np.int_(Rover.pos[1]), np.int_(Rover.pos[0])]
    Rover.worldmap_visited[rover_y-5:rover_y+5, rover_x-5:rover_x+5] += 1

    # 2) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain = color_thresh(warped, rgb_thresh_low=(160, 160, 160), rgb_thresh_high=(255, 255, 255))
    rocks = color_thresh(warped, rgb_thresh_low=(50, 50, 0), rgb_thresh_high=(255, 255, 10))
    obstacles = color_thresh(warped, rgb_thresh_low=(0, 0, 0), rgb_thresh_high=(160, 160, 160))

    # 3) Convert thresholded image pixel values to rover-centric coords and only retain those within the
    # field of view of the camera (+/- 45 degrees)
    terrain_x, terrain_y = rover_coords(terrain)
    terrain_x, terrain_y = pixels_in_fov(terrain_x, terrain_y)
    terrain_vision_image = image_coords(terrain_x, terrain_y, Rover.img.shape[:2])

    rocks_x, rocks_y = rover_coords(rocks)
    rocks_x, rocks_y = pixels_in_fov(rocks_x, rocks_y)
    rocks_vision_image = image_coords(rocks_x, rocks_y, Rover.img.shape[:2])

    obstacles_x, obstacles_y = rover_coords(obstacles)
    obstacles_x, obstacles_y = pixels_in_fov(obstacles_x, obstacles_y)
    obstacles_vision_image = image_coords(obstacles_x, obstacles_y, Rover.img.shape[:2])

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = obstacles_vision_image
    Rover.vision_image[:, :, 1] = rocks_vision_image
    Rover.vision_image[:, :, 2] = terrain_vision_image

    # 5) Convert rover-centric pixel values to world coordinates
    terrain_x_world, terrain_y_world = pix_to_world(
        terrain_x, terrain_y, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 2 * dst_size)

    rocks_x_world, rocks_y_world = pix_to_world(
        rocks_x, rocks_y, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 2 * dst_size)

    obstacles_x_world, obstacles_y_world = pix_to_world(
        obstacles_x, obstacles_y, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0],
        2 * dst_size)

    # 6) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacles_y_world, obstacles_x_world] = [255, 0, 0]
    Rover.worldmap[terrain_y_world, terrain_x_world] = [0, 0, 255]

    # For rocks, we draw a small white box around the rock center.
    rocks_y_center, rocks_x_center = [np.int_(np.median(rocks_y_world)), np.int_(np.median(rocks_x_world))]
    Rover.worldmap[rocks_y_center - 5:rocks_y_center + 5, rocks_x_center - 5:rocks_x_center + 5] = [255, 255, 255]


    # 7) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    rho, theta = to_polar_coords(terrain_x, terrain_y)
    Rover.nav_dists = rho
    Rover.nav_angles = theta

    return Rover
