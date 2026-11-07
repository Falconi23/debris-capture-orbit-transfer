"""
Phase 0: Hohmann transfer from 600 km to 610 km, then rendezvous with the debris.

Action: MultiDiscrete([3]) on the along-track axis (retrograde / coast / prograde, 400 N).
Observation: 9 values in [-1, 1]: SMA progress, eccentricity, altitude offset, radial and
along-track velocity, debris proximity and relative speed, mass fraction, along-track
direction to the debris.
Success: |SMA - r_target| < 1500 m, ecc < 0.005, debris distance < 1500 m, relative speed < 5 m/s.
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


class ADRTransferEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.mu = 3.986004418e14
        self.Re = 6371e3
        self.dt = 5.0               # s

        self.max_thrust   = 400.0   # N, along-track only
        self.initial_mass = 600.0   # kg
        self.dry_mass     = 450.0   # kg, leaves 150 kg of propellant
        self.Isp          = 300.0   # s
        self.g0           = 9.81    # m/s²

        self.r_park   = self.Re + 600e3   # 600 km
        self.r_target = self.Re + 610e3   # 610 km

        # Impulsive Hohmann reference, used only for logging
        a_t  = 0.5 * (self.r_park + self.r_target)
        v1   = np.sqrt(self.mu / self.r_park)
        v2   = np.sqrt(self.mu / self.r_target)
        self.dv1        = np.sqrt(self.mu * (2/self.r_park  - 1/a_t)) - v1
        self.dv2        = v2 - np.sqrt(self.mu * (2/self.r_target - 1/a_t))
        self.T_transfer = np.pi * np.sqrt(a_t**3 / self.mu)   # ≈ 2898 s

        self._angle_offset = -0.015   # rad

        # Phase the debris so it arrives at the transfer apoapsis together with the chaser
        omega2 = np.sqrt(self.mu / self.r_target**3)
        self._debris_angle0 = (self._angle_offset + np.pi) - omega2 * self.T_transfer

        # 0 = retrograde, 1 = coast, 2 = prograde. A Hohmann transfer is purely
        # along-track, so radial/normal thrust is left out of the action space.
        # A full burn costs ~0.68 kg per step, so 150 kg is ~220 burn steps.
        self.action_space = spaces.MultiDiscrete([3])

        self.observation_space = spaces.Box(low=-1., high=1., shape=(9,), dtype=np.float32)

        self.max_steps = 2000   # 10000 s, more than one orbit at 610 km (~5800 s)

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

    def _orbital_quality(self):
        """0 on the circular 600 km orbit, 1 on the circular 610 km orbit.

        Eccentricity only counts once the SMA has started to rise, so the first
        burn (which raises both) still scores positively.
        """
        a = self._sma()
        e = self._eccentricity()
        p = np.clip((a - self.r_park) / (self.r_target - self.r_park), -0.5, 1.5)
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

        # Chaser on the circular 600 km orbit
        ao = self._angle_offset
        v1 = np.sqrt(self.mu / self.r_park)
        self.r_chaser = np.array([self.r_park * np.cos(ao),
                                   self.r_park * np.sin(ao), 0.], dtype=np.float64)
        self.v_chaser = np.array([-v1 * np.sin(ao),
                                   v1 * np.cos(ao), 0.],           dtype=np.float64)

        # Debris on the circular 610 km orbit, phased for rendezvous
        a0 = self._debris_angle0
        v2 = np.sqrt(self.mu / self.r_target)
        self.r_debris = np.array([self.r_target * np.cos(a0),
                                   self.r_target * np.sin(a0), 0.], dtype=np.float64)
        self.v_debris = np.array([-v2 * np.sin(a0),
                                   v2 * np.cos(a0), 0.],           dtype=np.float64)

        self.prev_quality  = self._orbital_quality()
        self.prev_dist_deb = np.linalg.norm(self.r_chaser - self.r_debris)

        self.traj_x = [self.r_chaser[0]]
        self.traj_y = [self.r_chaser[1]]

        return self._get_obs(), {}

    def _get_obs(self):
        a   = self._sma()
        e   = self._eccentricity()
        r_n = np.linalg.norm(self.r_chaser)

        # 0 at 600 km, 1 at 610 km
        sma_p = np.clip((a - self.r_park) / (self.r_target - self.r_park), -0.5, 1.5)

        # Radial offset from r_target, ±2% mapped to ±1
        alt_frac = np.clip((r_n / self.r_target) - 1.0, -0.02, 0.02) / 0.02

        Q      = self.get_lvlh_matrix(self.r_chaser, self.v_chaser).T
        v_lvlh = Q @ self.v_chaser
        v_circ = np.sqrt(self.mu / r_n)

        # The sign of v_r tells where we are on the transfer ellipse: > 0 while
        # climbing, ~0 at apoapsis, which is when the second burn is due.
        v_r = np.clip(v_lvlh[0] / 3.0, -1., 1.)

        # Along-track excess over local circular speed
        v_t = np.clip((v_lvlh[1] - v_circ) / 5.0, -1., 1.)

        rel_pos      = self.r_debris - self.r_chaser
        dist_deb     = np.linalg.norm(rel_pos)
        vel_deb      = np.linalg.norm(self.v_chaser - self.v_debris)
        proximity    = 1. - np.clip(dist_deb / 15_000., 0., 1.)

        # Along-track component of the unit vector to the debris
        rel_lvlh     = Q @ rel_pos
        dist_safe    = max(dist_deb, 1e-6)
        dir_y        = np.clip(rel_lvlh[1] / dist_safe, -1., 1.)

        return np.array([
            np.clip(sma_p,       -1., 1.),
            np.clip(e * 50.,      0., 1.),    # saturates at ecc = 0.02
            alt_frac,
            v_r,
            v_t,
            proximity,
            np.clip(vel_deb/10.,  0., 1.),
            self.mass / self.initial_mass,
            dir_y,
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        thrust_map  = {0: -self.max_thrust, 1: 0., 2: self.max_thrust}
        agent_action = int(action[0])
        thrust_prog = thrust_map[agent_action]

        # LVLH column 1 is the along-track (V-bar) axis in ECI
        Q_to_eci   = self.get_lvlh_matrix(self.r_chaser, self.v_chaser)
        prograde   = Q_to_eci[:, 1]
        thrust_eci = prograde * thrust_prog

        thrust_mag = abs(thrust_prog)
        dm = (thrust_mag / (self.Isp * self.g0)) * self.dt
        self.mass = max(self.dry_mass, self.mass - dm)

        self.r_chaser, self.v_chaser = rk4_step_with_thrust(
            self.mu, self.r_chaser, self.v_chaser, thrust_eci / self.mass, self.dt)
        self.r_debris, self.v_debris = rk4_step_with_thrust(
            self.mu, self.r_debris, self.v_debris, np.zeros(3), self.dt)

        self.traj_x.append(self.r_chaser[0])
        self.traj_y.append(self.r_chaser[1])

        reward     = 0.
        terminated = False

        # A Hohmann transfer coasts between its two burns: thrusting after the
        # first 10 steps and before the 10-step window around apoapsis is penalized.
        target_apogee_step = int(self.T_transfer / self.dt)  # ~580

        if 10 < self.current_step < (target_apogee_step - 10):
            if agent_action != 1:
                reward -= 15.0
            else:
                reward += 1.0

        if self.mass <= self.dry_mass:
            reward -= 50.
            terminated = True

        # Change in orbital quality: large at each burn, zero while coasting
        quality           = self._orbital_quality()
        reward           += (quality - self.prev_quality) * 20.
        self.prev_quality  = quality

        # Approach progress, kept weak so it does not override the transfer
        dist_deb        = np.linalg.norm(self.r_chaser - self.r_debris)
        progress_d       = self.prev_dist_deb - dist_deb
        reward          += progress_d * 0.05
        self.prev_dist_deb = dist_deb

        reward -= dm * 0.5

        r_n = np.linalg.norm(self.r_chaser)
        if r_n < self.Re + 200e3 or r_n > self.Re + 2_000e3:
            reward -= 100.
            terminated = True

        a   = self._sma()
        e   = self._eccentricity()
        vel_deb = np.linalg.norm(self.v_chaser - self.v_debris)

        # Curriculum step 2 thresholds (step 1 trained with 5000 m)
        sma_ok = abs(a - self.r_target) < 1_500.
        ecc_ok = e < 0.005
        prx_ok = dist_deb < 1_500.
        vel_ok = vel_deb < 5.0

        if not terminated and sma_ok and ecc_ok and prx_ok and vel_ok:
            prop = self.mass - self.dry_mass
            print(f"  [{self.current_step:4d}] transfer complete  "
                  f"SMA={a/1e3:.3f}km  ecc={e:.6f}  "
                  f"dist={dist_deb:.0f}m  vel={vel_deb:.2f}m/s  "
                  f"prop={prop:.1f}kg")

            # Terminal bonus grows with leftover propellant
            reward    += 100. + prop * 0.5
            terminated = True

        truncated = self.current_step >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')
        ax.set_facecolor('#0d1117')

        ax.add_patch(plt.Circle((0, 0), self.Re, color='#1c6ea4', alpha=0.3))

        θ = np.linspace(0, 2*np.pi, 360)
        ax.plot(self.r_park   * np.cos(θ), self.r_park   * np.sin(θ),
                '--', color='#3fb950', alpha=0.6, label='Parking orbit (600 km)')
        ax.plot(self.r_target * np.cos(θ), self.r_target * np.sin(θ),
                '--', color='#79c0ff', alpha=0.6, label='Target orbit (610 km)')

        if len(self.traj_x) > 1:
            ax.plot(self.traj_x, self.traj_y,
                    color='#ff4444', lw=2.0, label='Transfer + approach (red)', zorder=3)
            ax.scatter(self.traj_x[0], self.traj_y[0],
                       color='#3fb950', marker='o', s=80, zorder=5, label='Start (600 km)')
            ax.scatter(self.traj_x[-1], self.traj_y[-1],
                       color='#e3b341', marker='^', s=120, zorder=5, label='Chaser (end)')

        ax.scatter(self.r_debris[0], self.r_debris[1],
                   color='magenta', marker='*', s=200, zorder=6, label='Debris (target)')

        ax.set_aspect('equal')
        all_vals = ([abs(x) for x in self.traj_x] +
                    [abs(y) for y in self.traj_y] +
                    [self.r_target])
        lim = max(all_vals) * 1.08
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

        dist_f = np.linalg.norm(self.r_chaser - self.r_debris)
        vel_f  = np.linalg.norm(self.v_chaser - self.v_debris)
        sma_f  = self._sma()
        ecc_f  = self._eccentricity()
        ax.set_title(
            f"Phase 0 — Transfer 600→610 km + Approach\n"
            f"dist={dist_f:.0f}m  vel={vel_f:.2f}m/s  "
            f"SMA={(sma_f-self.Re)/1e3:.2f}km  ecc={ecc_f:.5f}",
            color='white', pad=10, fontsize=9)
        ax.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
        ax.grid(color='white', ls='--', lw=0.4, alpha=0.2)
        ax.tick_params(colors='#8b949e')
        plt.show()

# Curriculum step 1: training from scratch, with the success thresholds in
# step() set to 5000 m. Kept for reference; swap with the step-2 block to run it.
'''if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    import os, random, torch
    import numpy as np
    import scipy.io as sio

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    env = ADRTransferEnv()
    env.reset(seed=SEED)

    d0 = np.linalg.norm(env.r_chaser - env.r_debris)
    print("Phase 0: transfer 600 -> 610 km")
    print(f"  Δv_1 = {env.dv1:.4f} m/s   Δv_2 = {env.dv2:.4f} m/s")
    print(f"  T_transfer = {env.T_transfer:.0f}s ({env.T_transfer/env.dt:.0f} step)")
    print(f"  Max Steps = {env.max_steps}")
    print(f"  Initial Distance = {d0/1e3:.1f} km\n")

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

    checkpoint_dir = "./phase0_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=2_000_000,
        save_path=checkpoint_dir,
        name_prefix="phase0"
    )

    print("Training for 10M steps")
    model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_callback
    )

    print("\nTraining done, running test episode")
    obs, _ = env.reset(seed=SEED)

    for i in range(env.max_steps):
        # Stochastic: the deterministic policy tends not to wait for the burn window
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            a   = env._sma()
            e   = env._eccentricity()
            r_n = np.linalg.norm(env.r_chaser)
            dist = np.linalg.norm(env.r_chaser - env.r_debris)
            vel  = np.linalg.norm(env.v_chaser - env.v_debris)

            print(f"\n[END OF FLIGHT — step {i+1}]")
            print(f"  SMA:          {a/1e3:.3f} km")
            print(f"  Eccentricity: {e:.6f}")
            print(f"  Altitude:     {(r_n-env.Re)/1e3:.3f} km")
            print(f"  Debris dist:  {dist:.0f} m")
            print(f"  Rel. vel:     {vel:.3f} m/s")
            print(f"  Propellant:   {env.mass-env.dry_mass:.1f} kg left")
            break

    # Explicit dtypes keep savemat from warning
    matlab_data = {
        'traj_x': np.array(env.traj_x, dtype=np.float64),
        'traj_y': np.array(env.traj_y, dtype=np.float64),
        'r_park': np.float64(env.r_park),
        'r_target': np.float64(env.r_target),
        'r_debris_final': np.array(env.r_debris, dtype=np.float64),
        'Re': np.float64(env.Re)
    }

    # Must match the file name loaded in plotp0.m
    mat_name = "phase0_orbits5000.mat"
    sio.savemat(mat_name, matlab_data)
    print(f"\nData exported for MATLAB to: {mat_name}")

    env.render()
    env.close()'''

# Curriculum step 2: resume from the latest step-1 checkpoint with the 1500 m thresholds.
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    import os, random, torch, glob
    import numpy as np
    import scipy.io as sio

    SEED = 42
    np.random.seed(SEED); random.seed(SEED)
    torch.manual_seed(SEED); set_random_seed(SEED)

    env = ADRTransferEnv()

    print("Phase 0: curriculum step 2\n")

    checkpoint_dir = "./phase0_checkpoints/"
    list_of_files = glob.glob(f"{checkpoint_dir}/*.zip")

    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        print(f"Loading {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
    else:
        print(f"No checkpoint found in {checkpoint_dir}; run step 1 first.")
        exit()

    print("\nResuming training")
    model.learn(
        total_timesteps=10_000_000,
        reset_num_timesteps=False,  # continue the step count from step 1
        callback=CheckpointCallback(
            save_freq=1_000_000,
            save_path=checkpoint_dir,
            name_prefix="phase0_step2"
        )
    )

    print("\nTest episode")
    obs, _ = env.reset(seed=SEED)

    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            a   = env._sma()
            e   = env._eccentricity()
            r_n = np.linalg.norm(env.r_chaser)
            dist = np.linalg.norm(env.r_chaser - env.r_debris)
            vel  = np.linalg.norm(env.v_chaser - env.v_debris)

            print(f"\n[END OF FLIGHT — step {i+1}]")
            print(f"  SMA:          {a/1e3:.3f} km   (target 610, ok if error < 1.5 km)")
            print(f"  Eccentricity: {e:.6f}   (target < 0.005)")
            print(f"  Altitude:     {(r_n-env.Re)/1e3:.3f} km")
            print(f"  Debris dist:  {dist:.0f} m     (target < 1500 m)")
            print(f"  Rel. vel:     {vel:.3f} m/s  (target < 5 m/s)")
            print(f"  Propellant:   {env.mass-env.dry_mass:.1f} kg left")
            break

    matlab_data = {
        'traj_x': np.array(env.traj_x, dtype=np.float64),
        'traj_y': np.array(env.traj_y, dtype=np.float64),
        'r_park': np.float64(env.r_park),
        'r_target': np.float64(env.r_target),
        'r_debris_final': np.array(env.r_debris, dtype=np.float64),
        'Re': np.float64(env.Re)
    }

    # Must match the file name loaded in plotp0.m
    mat_name = "phase0_orbits1500.mat"
    sio.savemat(mat_name, matlab_data)
    print(f"\nData exported for MATLAB to: {mat_name}")

    env.render()
    env.close()
