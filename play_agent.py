import time
from stable_baselines3 import DQN  # or replace with your agent class if using a different algorithm
from main import SnakeEnv         # Adjust this import if needed

def play_trained_agent(model_path="snake_dqn_model.zip"):
    # Create the environment
    env = SnakeEnv()
    # Load the trained model
    model = DQN.load(model_path, env=env)
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get action from the model (deterministic=True for a consistent policy)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Render the game state (this could be text or a graphical render)
        env.render()
        
        # Optional: slow down the loop so you can observe the game
        time.sleep(0.2)

    print("Episode finished with total reward:", total_reward)

if __name__ == "__main__":
    play_trained_agent()
