import gym 
from stable_baselines3 import PPO 
# from ExoskeletonEnv import ExoskeletonEnv
from ExoskeletonEnv2 import ExoskeletonEnv2
import os 

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# env = gym.make('BipedalWalker-v3')
# env = ExoskeletonEnv()
env = ExoskeletonEnv2()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 10000
TIMESTEPS = 100

for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

    # Render the environment
    obs = env.reset()
    done = False
    while not done:
        env.render()  # This will open a window displaying the environment
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    # Close the rendering window when done
    env.close()

    model.save(f"{models_dir}/{TIMESTEPS*i}")
