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
    HIP_HEIGHT = 0.2
    THIGH_LENGTH = 0.2
    CALF_LENGTH = 0.5
    MAX_STEPS = 100
    HIPS_DISTANCE = 0.2
    DELTA_GROUND = 0.05
    GROUND_POS = -0.8

    def __init__(self):
        # Define observation space and action space
        hip_low, hip_high = -0.2, 0.4
        knee_low, knee_high = -0.6, 0.2
        foot_low, foot_high = self.GROUND_POS, -0.2

        # Define the observation space as a Box with multiple dimensions
        observation_low = np.array([hip_low, knee_low, foot_low, hip_low, knee_low, foot_low])
        observation_high = np.array([hip_high, knee_high, foot_high, hip_high, knee_high, foot_high])
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        # self.observation_space = gym.spaces.Box(low=-1, high=0.4, shape=(6,), dtype=np.float32)
        print("observation space:", self.observation_space)
        # self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(3,), dtype=np.float32)

        self.nr_steps = 0
        self.max_steps = self.MAX_STEPS
        self.get_initial_positions()


        fig, ax = plt.subplots()
        self.ax = ax


    def get_initial_positions(self, step=1):
        self.hip1_x = -1
        self.hip1_y = 0
        
        self.hip2_x = self.hip1_x + self.HIPS_DISTANCE
        self.hip2_y = 0

        self.knee1_x = self.hip1_x + self.THIGH_LENGTH/2 * np.sin(step)
        self.knee1_y = self.hip1_y - self.HIP_HEIGHT - self.THIGH_LENGTH * np.cos(step)
        self.foot1_x = self.knee1_x 
        self.foot1_y = self.GROUND_POS

        self.knee2_x = self.knee1_x + self.HIPS_DISTANCE
        self.knee2_y = self.knee1_y - 0.1
        
        self.foot2_x = self.knee2_x 
        self.foot2_y = self.GROUND_POS

        self.foot = 0 #left

    def step(self, action, render=True):
        # Update exoskeleton state based on action
        self.old_state = [self.hip1_x, self.knee1_x, self.foot1_x, self.hip2_x, self.knee2_x, self.foot2_x]
        # hip1_torque, knee1_torque, foot1_torque, hip2_torque, knee2_torque, foot2_torque = action
        hip_torque, knee_torque, foot_torque = action

        # alternate between left and right foot:
        if self.nr_steps % 2 == 0:
            self.foot = 0 #left
        else:
            self.foot = 1 #right
        
        if self.foot==0:
            self.hip1_x += hip_torque
            self.hip1_y += hip_torque
            
            # it makes sense that when you move your hip you also move your knee:
            # self.knee1_x += knee1_torque
            self.knee1_x += knee_torque + hip_torque/2
            self.knee1_y += knee_torque / 2

            self.foot1_x += foot_torque
            self.foot1_y += foot_torque / 2
        else:
            self.hip2_x += hip_torque
            self.hip2_y += hip_torque

            # it makes sense that when you move your hip you also move your knee:
            self.knee2_x += knee_torque + hip_torque/2
            self.knee2_y += knee_torque / 2

            self.foot2_x += foot_torque
            self.foot2_y += foot_torque

        new_observation = [self.hip1_x, self.knee1_x, self.foot1_x, self.hip2_x, self.knee2_x, self.foot2_x]
        reward, done = self.calculate_reward(), self.nr_steps >= self.max_steps
        self.nr_steps += 1

        

        if render:
            self.render()
        
        self.draw_text("reward:" + str(reward))
        plt.pause(0.05)

        # also done when it falls down:
        if reward<0:
            print("it fell down!")
            self.ax.text(0.3, 0.3, "It fell down", size=50, 
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                        ec=(1., 0.5, 0.5),
                        fc=(1., 0.8, 0.8),
                        )
                )
            plt.draw()
            plt.pause(0.2)
            print("hip1_y, knee1_y, foot1_y:", self.hip1_y, self.knee1_y, self.foot1_y)
            print("hip2_y, knee2_y, foot2_y:", self.hip2_y, self.knee2_y, self.foot2_y)
            
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

    def calculate_reward(self):
        # Calculate reward based on balance criteria
        # the y-coordinates of the hips should 
        # be between -0.2 and 0.2 approx, knees between y=-0.6 and -0.2
        #  and foot between y=-0.8 and -0.5 .

        reward = 0

        hip_limits = [-0.4, 0.4]
        knee_limits = [-0.6, 0.2]
        foot_limits = [-0.8, -0.3]
        
        hip_balance = 0
        knee_balance = -0.4
        foot_balance = -0.6

        if self.nr_steps % 2 == 0:
            # big rewards for staying in the middle:
            if hip_balance -0.1 < self.hip1_y < hip_balance +0.1:
                reward += 30
            # medium rewards for staying in the range
            elif hip_limits[0] < self.hip1_y < hip_limits[1]:
                reward+= hip_balance - (hip_balance - self.hip1_y) * 10
            else:
                # negative rewards for falling:
                # reward  -= abs((hip_balance - self.hip1_y)) * 100
                reward  -= abs((hip_balance - self.hip1_y))* 10
        else:
            # big rewards for staying in the middle:
            if hip_balance -0.1 < self.hip2_y < hip_balance +0.1:
                reward += 30
            # medium rewards for staying in the range
            elif hip_limits[0] < self.hip2_y < hip_limits[1]:
                reward+= hip_balance - (hip_balance - self.hip2_y) * 10
            else:
                # negative rewards for falling:
                # reward  -= abs((hip_balance - self.hip1_y)) * 100
                reward  -= abs((hip_balance - self.hip2_y))* 10

       
        if self.nr_steps % 2 == 0:
            # big rewards for staying in the middle:
            if knee_balance -0.1 < self.knee1_y < knee_balance +0.1:
                reward += 30
            # medium rewards for staying in the range
            elif knee_limits[0] < self.knee1_y < knee_limits[1]:
                reward+= knee_balance - (knee_balance - self.knee1_y) * 10
            else:
                # negative rewards for falling:
                # reward  -= abs((knee_balance - self.knee1_y)) * 100
                reward  -= abs((knee_balance - self.knee1_y)) * 10
        else:
            # big rewards for staying in the middle:
            if knee_balance -0.1 < self.knee2_y < knee_balance +0.1:
                reward += 30
            # medium rewards for staying in the range
            elif knee_limits[0] < self.knee2_y < knee_limits[1]:
                reward+= knee_balance - (knee_balance - self.knee2_y) * 10
            else:
                # negative rewards for falling:
                # reward  -= abs((knee_balance - self.knee1_y)) * 100
                reward  -= abs((knee_balance - self.knee2_y)) * 10
            
        if self.nr_steps % 2 == 0:
            if self.foot1_y < self.GROUND_POS or self.foot1_y > foot_limits[1]:
                # negative rewards for falling:
                # reward  -= abs((foot_balance - self.foot1_y)) * 100
                # reward  -= abs((foot_balance - self.foot1_y)) * 10
                # how far it is from ground
                reward  -= abs((self.GROUND_POS - self.foot1_y)) * 100
        else:
            if self.foot2_y < self.GROUND_POS or self.foot2_y >  foot_limits[1]:
                # negative rewards for falling:
                # reward  -= abs((foot_balance - self.foot1_y)) * 100
                # how far it is from ground
                # reward  -= abs((foot_balance - self.foot2_y)) * 10
                reward  -= abs((self.GROUND_POS - self.foot2_y)) * 100


        hips_inclination = (self.hip2_y - self.hip1_y) / (self.hip2_x - self.hip1_x)
        degrees_inclination = math.degrees(math.atan(hips_inclination))

        if abs(degrees_inclination) < 35:
            reward += 100-degrees_inclination
        # else:
        #     # negative rewards for falling:
        #     reward -= 100

        print("-- hips_inclination degrees:", degrees_inclination)
        
        # TODO maybe: add distance between hips as reward:
        hips_distance = np.linalg.norm(self.hip1_x - self.hip2_x)

        if (hips_distance-self.HIPS_DISTANCE) > 7:
            reward -= 50
      
        
        #knee should not be above hip:
        if self.knee1_y > self.hip1_y:
            reward -= 100
        if self.knee2_y > self.hip2_y:
            reward -= 100

        # foot should not be in front of knee:
        if self.foot1_x > self.knee1_x:
            reward -= 100
        if self.foot2_x > self.knee2_x:
            reward -= 100

        # foot should not be above knee:
        if self.foot1_y > self.knee1_y:
            reward -= 100
        if self.foot2_y > self.knee2_y:
            reward -= 100

        # knees reverse pos:
        if self.knee1_x < self.hip1_x and self.foot1_x > self.knee1_x:
            reward -=200
        if self.knee2_x < self.hip2_x and self.foot2_x > self.knee2_x:
            reward -=200


        # at least 1 foot should be on the ground:
        if not(abs(self.foot1_y-self.GROUND_POS)<self.DELTA_GROUND or abs(self.foot2_y-self.GROUND_POS)<self.DELTA_GROUND):
            min_dist = min(abs(self.foot1_y-self.GROUND_POS)<self.DELTA_GROUND, abs(self.foot2_y-self.GROUND_POS)<self.DELTA_GROUND)
            reward -= 20+min_dist
            print("!!no foot on ground", abs(self.foot1_y-self.GROUND_POS), abs(self.foot2_y-self.GROUND_POS))
        

        if reward>0:
            # add movement on x as reward:
            if self.hip1_x > self.old_state[0]:
                reward  += (self.hip1_x  - self.old_state[0]) * 10
            if self.hip2_x > self.old_state[3]:
                reward  += (self.hip2_x  - self.old_state[3]) * 10

        return reward

    def reset(self):
        self.draw_text("RESET")
        # plt.pause(0.1)
        # Reset the environment to the initial state
        self.get_initial_positions()
        self.nr_steps = 0

        self._render_frame(self.nr_steps, self.ax)
        return [self.hip1_x, self.knee1_x, self.foot1_x, self.hip2_x, self.knee2_x, self.foot2_x]


    def render(self):
        # if self.render_mode == "rgb_array":
        return self._render_frame(self.nr_steps, self.ax)
    

    def _render_frame(self, i, ax):
        # plt.plot([100, 200, 300, 400, 500], [1, 2, 3, 4, 5], color="red")
        # plt.show()

        hip1_x, hip1_y, knee1_x, knee1_y, foot1_x, foot1_y, hip2_x, hip2_y, knee2_x, knee2_y, foot2_x, foot2_y = self.hip1_x, self.hip1_y, self.knee1_x, self.knee1_y, self.foot1_x, self.foot1_y, self.hip2_x, self.hip2_y, self.knee2_x, self.knee2_y, self.foot2_x, self.foot2_y

        ax.clear()
        ax.set_xlim(-1.5, 1)
        ax.set_ylim(-1, 0.5)

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

        ax.axhline(y=-0.8, color='g', linestyle='--', label='Ground')

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