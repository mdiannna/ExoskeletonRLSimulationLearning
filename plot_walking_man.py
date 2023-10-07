import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Constants for the person's body
HIP_HEIGHT = 0.2
THIGH_LENGTH = 0.2
CALF_LENGTH = 0.5
NUM_FRAMES = 100


# Function to calculate the positions of hip, knee, and foot
def update_position(step, hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip1_torque,knee1_torque, foot1_torque,
                        hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y, hip2_torque,knee2_torque, foot2_torque):
    # hip_x = 0
    # hip_y = 0
    
    # knee_x = hip_x - THIGH_LENGTH * np.sin(step)
    # knee_y = hip_y - HIP_HEIGHT - THIGH_LENGTH * np.cos(step)
    # foot_x = knee_x - CALF_LENGTH * np.sin(step)
    # foot_y = knee_y - CALF_LENGTH * np.cos(step)
    
    hip1_x+=hip1_torque
    hip1_y+=hip1_torque
    
    knee1_x+=knee1_torque
    knee1_y+=knee1_torque/2

    foot1_x+=foot1_torque
    foot1_y+=foot1_torque

    hip2_x += hip2_torque
    hip2_y += hip2_torque

    knee2_x += knee2_torque
    knee2_y += knee2_torque / 2

    foot2_x += foot2_torque
    foot2_y += foot2_torque

    return hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y


def get_initial_position(step=1):
    hip1_x = 0
    hip1_y = 0
    
    hip2_x = 0.2
    hip2_y = 0


    knee1_x = hip1_x + THIGH_LENGTH/2 * np.sin(step)
    knee1_y = hip1_y - HIP_HEIGHT - THIGH_LENGTH * np.cos(step)
    foot1_x = knee1_x 
    foot1_y = knee1_y - CALF_LENGTH * np.cos(step)


    knee2_x = knee1_x + 0.2
    knee2_y = knee1_y - 0.1
    
    foot2_x = knee2_x 
    foot2_y = knee2_y - CALF_LENGTH * np.cos(step)

    # knee_x = hip_x - THIGH_LENGTH * np.sin(step)
    # knee_y = hip_y - HIP_HEIGHT - THIGH_LENGTH * np.cos(step)
    # foot_x = knee_x - CALF_LENGTH * np.sin(step)
    # foot_y = knee_y - CALF_LENGTH * np.cos(step)

    return hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y



hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y = get_initial_position()

# Function to animate the walking motion
def animate(i):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 0.5)
    global hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y

    foot_length  = 0.2
    
    if i%2==0:
        hip1_torque = 0.06
        knee1_torque = 0.08
        foot1_torque = 0.02
        hip2_torque = -0.05
        knee2_torque = -0.07
        foot2_torque = -0.02
        
    else:
        hip2_torque = 0.06
        knee2_torque = 0.08
        foot2_torque = 0.02
        hip1_torque = -0.05
        knee1_torque = -0.07
        foot1_torque = -0.02
        
    hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y = update_position(i / 10, hip1_x, hip1_y, knee1_x, knee1_y,
                                         foot1_x, foot1_y, hip1_torque, knee1_torque, foot1_torque, hip2_x, hip2_y, 
                                         knee2_x, knee2_y, foot2_x, foot2_y,  hip2_torque, knee2_torque, foot2_torque)
    
    x_positions1 = [hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y]
    x_positions2 = [hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y]
    # x_positions = update_position(i / 10)  # Adjust the walking speed

    # Plot the body parts of the walking person - left leg
    ax.plot(x_positions1[0], x_positions1[1], 'bo', markersize=10)  # Hip
    
    ax.plot([x_positions1[0], x_positions1[2]], [x_positions1[1], x_positions1[3]], 'b-', linewidth=5)  # Thigh
    ax.plot([x_positions1[2], x_positions1[4]], [x_positions1[3], x_positions1[5]], 'b-', linewidth=5)  # Calf
    ax.plot(x_positions1[2], x_positions1[3], 'go', markersize=10)  # knee

    ax.plot(x_positions1[4], x_positions1[5], 'ro', markersize=10)  # Foot

    ax.plot([x_positions1[4], x_positions1[4] + foot_length/2], [x_positions1[5], x_positions1[5] ], 'b-', linewidth=5)  # Foot


    # Plot the body parts of the walking person - right leg
    ax.plot(x_positions2[0], x_positions2[1], 'bo', markersize=10)  # Hip
    
    ax.plot([x_positions2[0], x_positions2[2]], [x_positions2[1], x_positions2[3]], 'b-', linewidth=5)  # Thigh
    ax.plot([x_positions2[2], x_positions2[4]], [x_positions2[3], x_positions2[5]], 'b-', linewidth=5)  # Calf
    ax.plot(x_positions2[2], x_positions2[3], 'go', markersize=10)  # knee

    ax.plot(x_positions2[4], x_positions2[5], 'ro', markersize=10)  # Foot

    ax.plot([x_positions2[4], x_positions2[4] + foot_length/2], [x_positions2[5], x_positions2[5] ], 'b-', linewidth=5)  # Foot




# Create the Matplotlib figure and axis
fig, ax = plt.subplots()

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES, interval=1000)

# Show the animation
plt.show()
