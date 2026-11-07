"""
Phase 3: return transfer from 610 km to 600 km, carrying the captured debris (+100 kg).

Action: Box(3) in [-1, 1], cubed and scaled to ±400 N per LVLH axis.
Observation: 8 values in [-1, 1]: SMA descent progress, eccentricity, altitude error,
radial and along-track velocity, proximity to 600 km, speed excess, mass fraction.
Success: |SMA - r_park| < 250 m, ecc < 0.005, |r - r_park| < 250 m.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


def rk4_step_with_thrust(mu, r, v, thrust_acc_eci, dt):
    def deriv(r_, v_):
        return v_, -mu / np.linalg.norm(r_)**3 * r_ + thrust_acc_eci
    k1r, k1v = deriv(r, v)
    k2r, k2v = deriv(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
    k3r, k3v = deriv(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
    k4r, k4v = deriv(r +     dt*k3r, v +     dt*k3v)
    return (r + dt/6*(k1r+2*k2r+2*k3r+k4r),
            v + dt/6*(k1v+2*k2v+2*k3v+k4v))


class ADRReturnEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.mu  = 3.986004418e14
        self.Re  = 6371e3
        self.dt  = 5.0               # s

        self.max_main_thrust = 400.0  # N

        # ~70 kg of propellant assumed left after Phases 0-2
        self.initial_mass = 620.0    # kg
        self.dry_mass     = 550.0    # kg, chaser 450 + debris 100
        self.Isp          = 300.0    # s
        self.g0           = 9.81     # m/s²

        # Names kept from Phase 0: here r_target is the departure orbit, r_park the arrival
        self.r_target = self.Re + 610e3
        self.r_park   = self.Re + 600e3

        a_t    = 0.5 * (self.r_park + self.r_target)
        v_t     = np.sqrt(self.mu / self.r_target)
        v_p     = np.sqrt(self.mu / self.r_park)
        self.dv_ret1    = v_t - np.sqrt(self.mu * (2/self.r_target - 1/a_t))
        self.dv_ret2    = np.sqrt(self.mu * (2/self.r_park  - 1/a_t)) - v_p
        self.T_transfer = np.pi * np.sqrt(a_t**3 / self.mu)  # ~2886 s, ~577 steps

        self.action_space = spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-1., high=1., shape=(8,), dtype=np.float32)

        self.max_steps = 2000

    def _sma(self):
        r   = np.linalg.norm(self.r_chaser)
        v   = np.linalg.norm(self.v_chaser)
        eps = 0.5*v**2 - self.mu/r
        return -self.mu/(2*eps) if eps < 0 else 1e12

    def _eccentricity(self):
        r_n   = np.linalg.norm(self.r_chaser)
        v_n   = np.linalg.norm(self.v_chaser)
        e_vec = ((v_n**2 - self.mu/r_n)*self.r_chaser
                 - np.dot(self.r_chaser, self.v_chaser)*self.v_chaser) / self.mu
        return np.linalg.norm(e_vec)

    def _orbital_quality_descent(self):
        """0 on the circular 610 km orbit, 1 on the circular 600 km orbit.

        Mirror of the Phase 0 metric: eccentricity only counts once the SMA has
        started to drop, so the first retrograde burn still scores positively.
        """
        a   = self._sma()
        e   = self._eccentricity()
        p   = np.clip((self.r_target - a) / (self.r_target - self.r_park), -0.5, 1.5)
        return p - e * 100.0 * max(0., p)

    def get_lvlh_matrix(self, r_ref, v_ref):
        r_hat = r_ref / np.linalg.norm(r_ref)
        h_vec = np.cross(r_ref, v_ref)
        h_hat = h_vec / np.linalg.norm(h_vec)
        v_hat = np.cross(h_hat, r_hat)
        return np.column_stack((r_hat, v_hat, h_hat))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.mass         = self.initial_mass
        self.current_step = 0

        # Hand-over point from Phase 2: small angular offset, sign set by the seed parity
        ao = +0.015 if (seed is None or seed % 2 == 0) else -0.015
        v_t = np.sqrt(self.mu / self.r_target)
        self.r_chaser = np.array([self.r_target * np.cos(ao),
                                   self.r_target * np.sin(ao), 0.], dtype=np.float64)
        self.v_chaser = np.array([-v_t * np.sin(ao),
                                   v_t * np.cos(ao), 0.],           dtype=np.float64)

        # Residual velocity error from Phase 2, up to ±0.1 m/s per LVLH axis
        rng = np.random.default_rng(seed)
        delta_v_lvlh = rng.uniform(-0.1, 0.1, size=3)
        Q_to_eci     = self.get_lvlh_matrix(self.r_chaser, self.v_chaser)
        self.v_chaser += Q_to_eci @ delta_v_lvlh

        # Unpowered reference point on the 600 km orbit, exported for the plots
        v_p = np.sqrt(self.mu / self.r_park)
        self.r_beacon = np.array([self.r_park, 0., 0.], dtype=np.float64)
        self.v_beacon = np.array([0., v_p, 0.],          dtype=np.float64)

        self.prev_quality = self._orbital_quality_descent()

        self.traj_x = [self.r_chaser[0]]
        self.traj_y = [self.r_chaser[1]]

        return self._get_obs(), {}

    def _get_obs(self):
        a   = self._sma()
        e   = self._eccentricity()
        r_n = np.linalg.norm(self.r_chaser)

        # 0 at 610 km, 1 at 600 km
        sma_descent = np.clip((self.r_target - a) / (self.r_target - self.r_park),
                              -0.5, 1.5)

        # 0 at 600 km, 1 at 610 km
        alt_err = np.clip((r_n - self.r_park) / (self.r_target - self.r_park),
                          -0.5, 1.5)

        Q      = self.get_lvlh_matrix(self.r_chaser, self.v_chaser).T
        v_lvlh = Q @ self.v_chaser
        v_circ_current = np.sqrt(self.mu / r_n)
        v_circ_park    = np.sqrt(self.mu / self.r_park)

        v_r = v_lvlh[0] / 5.0
        v_t = (v_lvlh[1] - v_circ_park) / 5.0            # vs circular speed at 600 km

        # 1 at or below 600 km, 0 from 10 km above
        alt_excess  = max(0., r_n - self.r_park)
        proximity   = 1. - np.clip(alt_excess / 10_000., 0., 1.)

        vel_excess  = abs(np.linalg.norm(self.v_chaser) - v_circ_park)

        obs = np.array([
            np.clip(sma_descent,   -1., 1.),
            np.clip(e * 100.,       0., 1.),
            np.clip(alt_err,       -1., 1.),
            np.clip(v_r,           -1., 1.),
            np.clip(v_t,           -1., 1.),
            proximity,
            np.clip(vel_excess/10., 0., 1.),
            self.mass / self.initial_mass,
        ], dtype=np.float32)

        return obs

    def step(self, action):
        self.current_step += 1

        # Cubing gives finer control near zero thrust
        action_s    = np.power(np.clip(action, -1., 1.), 3)
        thrust_lvlh = action_s * self.max_main_thrust
        thrust_mag  = np.linalg.norm(thrust_lvlh)

        dm = (thrust_mag / (self.Isp * self.g0)) * self.dt
        self.mass = max(self.dry_mass, self.mass - dm)

        Q_to_eci   = self.get_lvlh_matrix(self.r_chaser, self.v_chaser)
        thrust_eci = Q_to_eci @ thrust_lvlh
        acc_eci    = thrust_eci / self.mass

        self.r_chaser, self.v_chaser = rk4_step_with_thrust(
            self.mu, self.r_chaser, self.v_chaser, acc_eci, self.dt)
        self.r_beacon, self.v_beacon = rk4_step_with_thrust(
            self.mu, self.r_beacon, self.v_beacon, np.zeros(3), self.dt)

        self.traj_x.append(self.r_chaser[0])
        self.traj_y.append(self.r_chaser[1])

        reward    = 0.
        terminated = False

        if self.mass <= self.dry_mass:
            reward    -= 100.
            terminated = True

        # Reward the change in SMA/eccentricity rather than distance to a point
        # below the chaser: that "ghost target" distance stays constant while
        # coasting at 610 km, so it gave no gradient and the agent never descended.
        quality  = self._orbital_quality_descent()
        dq       = quality - self.prev_quality
        reward  += dq * 20.
        self.prev_quality = quality

        a = self._sma()
        e = self._eccentricity()
        r_n = np.linalg.norm(self.r_chaser)

        # alt_ok as well, since SMA alone is satisfied anywhere on a 600 km-SMA ellipse
        sma_ok = abs(a - self.r_park) < 250.
        ecc_ok = e < 0.005
        alt_ok = abs(r_n - self.r_park) < 250.

        if sma_ok:
            # Circularization bonus once the SMA is right, to drive the second burn
            reward += (0.001 - e) * 10. if ecc_ok else 0.

        reward -= dm * 0.05

        r_n = np.linalg.norm(self.r_chaser)
        if r_n < self.Re + 100e3 or r_n > self.Re + 2_000e3:
            reward    -= 100.
            terminated = True

        if not terminated and sma_ok and ecc_ok and alt_ok:
            v_circ = np.sqrt(self.mu / r_n)
            vel_excess = abs(np.linalg.norm(self.v_chaser) - v_circ)
            prop = self.mass - self.dry_mass
            print(f"  [{self.current_step:4d}] return complete  "
                  f"SMA={a/1e3:.2f}km  ecc={e:.6f}  "
                  f"vel_excess={vel_excess:.3f}m/s  "
                  f"prop_left={prop:.1f}kg")
            reward    += 150.
            terminated = True

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')

        ax.add_patch(plt.Circle((0, 0), self.Re, color='#1c6ea4', alpha=0.3))

        θ = np.linspace(0, 2*np.pi, 360)
        ax.plot(self.r_target * np.cos(θ), self.r_target * np.sin(θ),
                '--', color='#79c0ff', alpha=0.6, label='Debris orbit (610 km)')
        ax.plot(self.r_park   * np.cos(θ), self.r_park   * np.sin(θ),
                '--', color='#3fb950', alpha=0.6, label='Safe zone (600 km)')

        if len(self.traj_x) > 1:
            ax.plot(self.traj_x, self.traj_y,
                    color='#ffcc00', lw=2.5, label='Return with load', zorder=3)
            ax.scatter(self.traj_x[0],  self.traj_y[0],
                       color='magenta', marker='*', s=160, zorder=5,
                       label='Pick-up point (departure)')
            ax.scatter(self.traj_x[-1], self.traj_y[-1],
                       color='#e3b341', marker='s', s=120, zorder=5,
                       label='Chaser at the base (end)')

        ax.set_aspect('equal')
        margin = 500_000
        ax.set_xlim(min(self.traj_x) - margin, max(self.traj_x) + margin)
        ax.set_ylim(min(self.traj_y) - margin, max(self.traj_y) + margin)

        ax.set_title("Phase 3 — Return journey 610 km → 600 km (including debris)",
                     color='white', pad=12)
        ax.legend(facecolor='#161b22', labelcolor='white')
        ax.grid(color='white', ls='--', lw=0.4, alpha=0.2)
        ax.tick_params(colors='#8b949e')
        plt.show()


if __name__ == "__main__":
    import os, random, torch
    import numpy as np
    import scipy.io as sio
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback

    env = ADRReturnEnv()

    # Curriculum: each step resumes from the previous step's checkpoint with
    # tighter success tolerances in step()
    model_path = "./phase3_checkpoints/phase3_resumed_18000000_steps.zip"

    print(f"Loading {model_path}")

    model = PPO.load(model_path, env=env)

    checkpoint_dir = "./phase3_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=checkpoint_dir,
        name_prefix="phase3_resumed"
    )

    NEW_TIMESTEPS = 10_000_000

    print(f"Resuming training for {NEW_TIMESTEPS} more steps")

    model.learn(
        total_timesteps=NEW_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=False  # continue the step count from the checkpoint
    )

    print("\nTraining done, test episode")
    obs, _ = env.reset()
    
    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    matlab_data = {
        'traj_x': np.array(env.traj_x, dtype=np.float64),
        'traj_y': np.array(env.traj_y, dtype=np.float64),
        'r_park': np.float64(env.r_park),
        'r_target': np.float64(env.r_target),
        'r_beacon_final': np.array(env.r_beacon, dtype=np.float64),
        'Re': np.float64(env.Re)
    }
    
    # One file per curriculum step; must match the file name loaded in plotp3.m
    mat_name = "phase3_curriculum_step250.mat" 
    sio.savemat(mat_name, matlab_data)
    print(f"Data exported for MATLAB to '{mat_name}'")
    env.render()
    env.close()


# First curriculum step, training from scratch (kept for reference).
'''if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    import os, random, torch

    SEED = 42
    np.random.seed(SEED); random.seed(SEED)
    torch.manual_seed(SEED); set_random_seed(SEED)

    env = ADRReturnEnv()
    env.reset(seed=SEED)

    print("Phase 3: return 610 -> 600 km")
    print(f"  Δv_ret1 ≈ {env.dv_ret1:.3f} m/s  Δv_ret2 ≈ {env.dv_ret2:.3f} m/s")
    print(f"  T_transfer ≈ {env.T_transfer:.0f} s  ({env.T_transfer/env.dt:.0f} step)")
    print(f"  Mass (including debris): {env.initial_mass:.0f} kg  "
          f"(dry {env.dry_mass:.0f} kg + {env.initial_mass-env.dry_mass:.0f} kg prop)")
    print(f"  Success: |SMA − 600km| < 5km  AND  ecc < 0.003\n")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        gamma=0.995,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=SEED,
    )

    os.makedirs("./phase3_checkpoints/", exist_ok=True)
    model.learn(
        total_timesteps=3_000_000,
        callback=CheckpointCallback(
            save_freq=1_000_000,
            save_path="./phase3_checkpoints/",
            name_prefix="phase3"
        )
    )

    print("\nTest episode")
    obs, _ = env.reset(seed=SEED)
    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.render()
    env.close()'''