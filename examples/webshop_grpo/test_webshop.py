import sys
sys.path.insert(0, "/workspace/WebShop")

import gym
from web_agent_site.envs import WebAgentTextEnv

# Test environment creation
print("Creating environment...")
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100)

print("Resetting environment...")
obs = env.reset()
print(f"Observation type: {type(obs)}")
print(f"Observation preview: {obs[:200]}")

print("\nTesting search action...")
try:
    obs, reward, done, info = env.step("search[red t-shirt]")
    print(f"Search result type: {type(obs)}")
    print(f"Search result preview: {obs[:200]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting click action...")
try:
    obs, reward, done, info = env.step("click[B001]")
    print(f"Click result type: {type(obs)}")
    print(f"Click result preview: {obs[:200]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
