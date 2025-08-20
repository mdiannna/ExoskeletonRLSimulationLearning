import gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from gym import Env, spaces
import math

# import matplotlib as mpl
# mpl.use('Agg')

# Define the RL environment
class ExoskeletonEnv2(gym.Env):
    HIP_HEIGHT = 0.8
    THIGH_LENGTH = 0.4
    DELTA_THIGH_ERR = 0.1
    CALF_LENGTH = 0.4
    # MAX_STEPS = 100
    MAX_STEPS = 20 #stop if learned to do 20 steps without falling
    HIPS_DISTANCE = 0.2
    DELTA_GROUND = 0.05
    # GROUND_POS = -0.8
    GROUND_POS = 0

    def __init__(self):
        # Define observation space and action space
        hip_low, hip_high = -0.2, 0.4
        knee_low, knee_high = 0, 0.2
        foot_low, foot_high = self.GROUND_POS, 0.6

        # Define the observation space as a Box with multiple dimensions
        # observation_low = np.array([hip_low, knee_low, foot_low, hip_low, knee_low, foot_low])
        # observation_high = np.array([hip_high, knee_high, foot_high, hip_high, knee_high, foot_high])
        # self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        obs_dim = 24  # 6 joints * (x + y + vx + vy)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # self.observation_space = gym.spaces.Box(low=-1, high=0.4, shape=(6,), dtype=np.float32)
        print("observation space:", self.observation_space)
        # self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) #positive action space for moving forward

        self.nr_steps = 0
        self.max_steps = self.MAX_STEPS
        self.get_initial_positions()


        fig, ax = plt.subplots()
        self.ax = ax


    def get_initial_positions(self):
        self.hip1_x = -1
        self.hip1_y = self.HIP_HEIGHT
        self.hip2_x = self.hip1_x + self.HIPS_DISTANCE
        self.hip2_y = self.HIP_HEIGHT

        self.knee1_x = self.hip1_x
        self.knee1_y = self.hip1_y - self.THIGH_LENGTH
        self.foot1_x = self.knee1_x
        self.foot1_y = self.GROUND_POS

        self.knee2_x = self.hip2_x
        self.knee2_y = self.hip2_y - self.THIGH_LENGTH
        self.foot2_x = self.knee2_x
        self.foot2_y = self.GROUND_POS

        self.foot = 0  # left

        # initialize old positions for velocity calculation
        self.hip1_y_old = self.hip1_y
        self.knee1_y_old = self.knee1_y
        self.foot1_y_old = self.foot1_y
        self.hip2_y_old = self.hip2_y
        self.knee2_y_old = self.knee2_y
        self.foot2_y_old = self.foot2_y

    def step(self, action, render=True):
        # Save old state
        self.old_state = [self.hip1_x, self.knee1_x, self.foot1_x, self.hip2_x, self.knee2_x, self.foot2_x]

        print("action:", action)
        
        hip_torque, knee_torque, foot_torque = action * 2
        self.foot = 0 if self.nr_steps % 2 == 0 else 1
        dt = 0.3  # bigger timestep for more movement

        # Scaling factors for more pronounced movement
        hip_scale = 0.2
        knee_scale = 0.5
        foot_scale = 0.4

        if self.foot == 0:
            self.hip1_x += hip_torque * dt * hip_scale
            self.hip1_y += hip_torque * dt * hip_scale / 2

            self.knee1_x += (knee_torque + hip_torque/2) * dt * knee_scale
            self.knee1_y += knee_torque * dt 

            self.foot1_x += foot_torque * dt * foot_scale * 2
            self.foot1_y += foot_torque * dt / 2
        else:
            self.hip2_x += hip_torque * dt * hip_scale
            self.hip2_y += hip_torque * dt * hip_scale / 2

            self.knee2_x += (knee_torque + hip_torque/2) * dt * knee_scale
            self.knee2_y += knee_torque * dt 

            self.foot2_x += foot_torque * dt * foot_scale * 2
            self.foot2_y += foot_torque * dt / 2
            
        # new observation with the full state:
        new_observation = [
            # hip1
            self.hip1_x, self.hip1_y,
            (self.hip1_x - self.old_state[0]) / dt,  # velocity x
            (self.hip1_y - getattr(self, 'hip1_y_old', self.hip1_y)) / dt,  # velocity y

            # knee1
            self.knee1_x, self.knee1_y,
            (self.knee1_x - self.old_state[1]) / dt,
            (self.knee1_y - getattr(self, 'knee1_y_old', self.knee1_y)) / dt,

            # foot1
            self.foot1_x, self.foot1_y,
            (self.foot1_x - self.old_state[2]) / dt,
            (self.foot1_y - getattr(self, 'foot1_y_old', self.foot1_y)) / dt,

            # hip2
            self.hip2_x, self.hip2_y,
            (self.hip2_x - self.old_state[3]) / dt,
            (self.hip2_y - getattr(self, 'hip2_y_old', self.hip2_y)) / dt,

            # knee2
            self.knee2_x, self.knee2_y,
            (self.knee2_x - self.old_state[4]) / dt,
            (self.knee2_y - getattr(self, 'knee2_y_old', self.knee2_y)) / dt,

            # foot2
            self.foot2_x, self.foot2_y,
            (self.foot2_x - self.old_state[5]) / dt,
            (self.foot2_y - getattr(self, 'foot2_y_old', self.foot2_y)) / dt,
        ]

        # Compute reward
        reward = self.calculate_reward()
        done = self.nr_steps >= self.max_steps
        self.nr_steps += 1

        # Render if requested
        if render:
            self.render()
            self.draw_text(f"reward: {reward:.2f}")
            self.draw_text_lower(f"step: {self.nr_steps}")

            plt.pause(0.05)
        
        self.hip1_y_old = self.hip1_y
        self.knee1_y_old = self.knee1_y
        self.foot1_y_old = self.foot1_y
        self.hip2_y_old = self.hip2_y
        self.knee2_y_old = self.knee2_y
        self.foot2_y_old = self.foot2_y

        # if reward<-10:
        #     done = True
        if reward<0:
            done = True

        return new_observation, reward, done, {}

    def draw_text(self, text):
        self.ax.text(0.35, 0.35, text, size=20, 
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=(0., 0.5, 0.5),
                    fc=(0., 0.8, 0.8),
                    )
            )
        plt.draw()
        plt.pause(0.4)

    def draw_text_lower(self, text):
        self.ax.text(
            0.98, 0.02, text, size=10,
            ha='right', va='bottom',          # anchor text to lower right
            transform=self.ax.transAxes,      # use axes coords
            bbox=dict(
                boxstyle="round",
                ec=(0., 0.5, 0.5),
                fc=(0., 0.8, 0.8),
            )
        )
        plt.draw()
        plt.pause(0.4)

    def calculate_reward(self):
        reward = 0.0

        # --- Balance reward ---
        hip_target = 0.0
        knee_target = -0.4
        foot_target = self.GROUND_POS

        # Hip balance
        if self.foot == 0:
            hip_y = self.hip1_y
            knee_y = self.knee1_y
            knee_x = self.knee1_x
            foot_y = self.foot1_y
            foot_x = self.foot1_x
            hip_x_prev = self.old_state[0]
            hip_x_curr = self.hip1_x
        else:
            hip_y = self.hip2_y
            knee_y = self.knee2_y
            knee_x = self.knee2_x
            foot_x = self.foot2_x
            foot_y = self.foot2_y
            hip_x_prev = self.old_state[3]
            hip_x_curr = self.hip2_x

        # Reward for keeping hips/knees in reasonable range
        reward += max(0, 1 - abs(hip_y - hip_target)) * 2
        reward += max(0, 1 - abs(knee_y - knee_target)) * 1.5

        # # Reward for foot near ground
        foot_dist = abs(foot_y - foot_target)
        reward += max(0, 1 - foot_dist / 0.2) * 1.0


        # Forward movement reward
        fwd_movement_coeff = 20.0
        if hip_x_curr > hip_x_prev: #only reward forward movement without penalizing bckwd movement
            reward += (hip_x_curr - hip_x_prev) * fwd_movement_coeff

        # Penalties for unrealistic positions
        if knee_y > hip_y:
            reward -= 2.0
        if foot_y > knee_y:
            reward -= 2.0
        
        # if foot_y < self.GROUND_POS - 0.2:
        #     reward -= 3.0
        
        # knee should not be in behind foot and hip:
        if foot_x > knee_x and knee_x<hip_x_curr:
            reward -= 100

        if foot_x > knee_x:
            reward -= 10

        # reward based on thigh length:
        calculated_thigh_length = math.sqrt((knee_x - hip_x_curr)**2 + (knee_y - hip_y)**2)
        reward-= abs(calculated_thigh_length-self.THIGH_LENGTH) - self.DELTA_THIGH_ERR
        
        # Hips inclination penalty
        try:
            hips_inclination = (self.hip2_y - self.hip1_y) / (self.hip2_x - self.hip1_x)
            degrees_inclination = math.degrees(math.atan(hips_inclination))
            reward += max(0, 1 - abs(degrees_inclination) / 45) * 2
        except ZeroDivisionError:
            reward -= 1.0

        return reward

    def reset(self):
        self.draw_text_lower("RESET")
        plt.pause(0.1)
        # Reset the environment to the initial state
        self.get_initial_positions()
        self.nr_steps = 0

        self._render_frame(self.nr_steps, self.ax)
        # return [self.hip1_x, self.knee1_x, self.foot1_x, self.hip2_x, self.knee2_x, self.foot2_x]
        return np.zeros(24, dtype=np.float32)

    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame(self.nr_steps, self.ax)
    

    def _render_frame(self, i, ax):
        # plt.plot([100, 200, 300, 400, 500], [1, 2, 3, 4, 5], color="red")
        # plt.show()

        hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y = self.hip1_x, self.hip1_y, self.knee1_x, self.knee1_y, self.foot1_x, self.foot1_y, self.hip2_x, self.hip2_y, self.knee2_x, self.knee2_y, self.foot2_x, self.foot2_y

        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1, 0.5) #TODO: check if neeed

        foot_length  = 0.2            
           
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

        ax.axhline(y=self.GROUND_POS, color='g', linestyle='--', label='Ground')

        plt.draw()

        plt.pause(0.1)
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     pygame.display.init()
        #     self.window = pygame.display.set_mode((self.window_size, self.window_size))
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size, self.window_size))
        # canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # if self.render_mode == "human":
        #     # The following line copies our drawings from `canvas` to the visible window
        #     self.window.blit(canvas, canvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()

        #     # We need to ensure that human-rendering occurs at the predefined framerate.
        #     # The following line will automatically add a delay to keep the framerate stable.
        #     self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )

        # ax.show()