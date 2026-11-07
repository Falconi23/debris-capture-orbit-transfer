import time
from stable_baselines3 import PPO

# Needs Phase12.py in the same folder
from Phase12 import ADRFullEnv

print("\nInitializing environment and network")
env = ADRFullEnv()
obs, _ = env.reset(seed=42)

# Untrained policy with the Phase 1-2 architecture: timing does not depend on the weights
model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), verbose=0)

print("Warming up")
for _ in range(100):
    model.predict(obs, deterministic=True)

print("Running 10,000 timing iterations")
start_time = time.perf_counter()

for _ in range(10000):
    model.predict(obs, deterministic=True)

end_time = time.perf_counter()

average_time_ms = ((end_time - start_time) / 10000) * 1000

print(f"\nAverage time per step: {average_time_ms:.5f} ms\n")
