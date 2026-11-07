"""
Phases 1-2: 6-DOF capture of the debris and return to the start point, with an
obstacle in between. Phases 0 and 3 (600 <-> 700 km Hohmann legs) are scripted
around the learned episode so the full mission can be plotted.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def q_to_rot_mat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy],
        [    2*qx*qy + 2*qw*qz, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qw*qx],
        [    2*qx*qz - 2*qw*qy,     2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float64)

def q_dot(q, omega):
    qw, qx, qy, qz = q
    wx, wy, wz = omega
    w_mat = np.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0]
    ], dtype=np.float64)
    return 0.5 * (w_mat @ q)

def rk4_step(mu, r, v, dt):
    """Unpowered two-body RK4 step, used for the scripted Phases 0 and 3."""
    def deriv(r_, v_):
        return v_, -mu / np.linalg.norm(r_)**3 * r_
    k1r, k1v = deriv(r, v)
    k2r, k2v = deriv(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
    k3r, k3v = deriv(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
    k4r, k4v = deriv(r +     dt*k3r, v +     dt*k3v)
    r_new = r + (dt/6.) * (k1r + 2*k2r + 2*k3r + k4r)
    v_new = v + (dt/6.) * (k1v + 2*k2v + 2*k3v + k4v)
    return r_new, v_new

def circular_state(r_orbit, angle_rad, mu):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    v_c  = np.sqrt(mu / r_orbit)
    return (r_orbit * np.array([c, s, 0.], dtype=np.float64),
            v_c     * np.array([-s, c, 0.], dtype=np.float64))


class ADRFullEnv(gym.Env):
    def __init__(self):
        super(ADRFullEnv, self).__init__()

        self.mu = 3.986004418e14 
        self.Re = 6371e3         
        self.dt = 1.0            
        self.max_thrust = 50.0   
        self.max_torque = 5.0    
        
        self.initial_mass = 600.0   
        self.dry_mass = 450.0       
        self.debris_mass = 100.0    
        self.Isp = 220.0            
        self.g0 = 9.81              
        self.base_I = np.array([50.0, 50.0, 50.0]) 
        
        # Scripted Hohmann legs (Phases 0 and 3)
        self.dt_rk4 = 5.0
        self.r_park = self.Re + 600e3  # departure orbit
        self.r_target = self.Re + 700e3 # debris orbit
        
        a_t = 0.5 * (self.r_park + self.r_target)
        v1 = np.sqrt(self.mu / self.r_park)
        v2 = np.sqrt(self.mu / self.r_target)
        self.dv1 = np.sqrt(self.mu * (2/self.r_park - 1/a_t)) - v1
        self.dv2 = v2 - np.sqrt(self.mu * (2/self.r_target - 1/a_t))
        self.T_transfer = np.pi * np.sqrt(a_t**3 / self.mu)

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        
        # Last entry is the phase flag, bounded at 3.0 to cover Phases 0-3
        high_list = [20.0]*3 + [5.0]*3 + [20.0]*3 + [5.0]*3 + [1.0]*4 + [1.0] + [3.0]
        self.observation_space = spaces.Box(
            low=np.array(high_list)*-1.0, 
            high=np.array(high_list), 
            dtype=np.float32
        )
        
        self.ost_r = 2.0 
        self.goal_radius = 0.5
        
        # ECI trajectory of the scripted legs, tagged with the phase for plotting
        self.traj_eci_phase03 = []
        self.traj_phase_log_03 = []
        self.mission_done = False

    def get_gravity(self, r_vec):
        r_norm = np.linalg.norm(r_vec)
        return -self.mu / (r_norm**3) * r_vec

    def get_lvlh_matrix(self, r_ref, v_ref):
        r_hat = r_ref / np.linalg.norm(r_ref)
        h_vec = np.cross(r_ref, v_ref)
        h_hat = h_vec / np.linalg.norm(h_vec)
        v_hat = np.cross(h_hat, r_hat)
        return np.column_stack((r_hat, v_hat, h_hat))

    def _scripted_hohmann_forward(self):
        """Phase 0: impulsive prograde burn, RK4 coast to apoapsis.

        Propellant for both burns is charged here via the rocket equation.
        """
        r_curr, v_curr = circular_state(self.r_park, 0.0, self.mu)
        self.traj_eci_phase03 = [r_curr.copy()]
        self.traj_phase_log_03 = [0]
        
        v_hat = v_curr / np.linalg.norm(v_curr)
        v_curr += abs(self.dv1) * v_hat
        
        n_steps_h = int(self.T_transfer / self.dt_rk4) + 1
        for _ in range(n_steps_h):
            r_curr, v_curr = rk4_step(self.mu, r_curr, v_curr, self.dt_rk4)
            self.traj_eci_phase03.append(r_curr.copy())
            self.traj_phase_log_03.append(0)
            
        dm_min = self.mass * (1.0 - np.exp(-(abs(self.dv1)+abs(self.dv2)) / (self.Isp * self.g0)))
        self.mass -= dm_min

    def _scripted_hohmann_return(self):
        """Phase 3: impulsive retrograde burn, RK4 coast down to 600 km.

        Returns the completion bonus.
        """
        r_curr, v_curr = self.r_chaser.copy(), self.v_chaser.copy()

        v_hat = v_curr / np.linalg.norm(v_curr)
        v_curr -= abs(self.dv2) * v_hat # current point becomes the apoapsis
        
        n_steps_h = int(self.T_transfer / self.dt_rk4) + 1
        for _ in range(n_steps_h):
            r_curr, v_curr = rk4_step(self.mu, r_curr, v_curr, self.dt_rk4)
            self.traj_eci_phase03.append(r_curr.copy())
            self.traj_phase_log_03.append(3)
            
        dm_min = self.mass * (1.0 - np.exp(-(abs(self.dv1)+abs(self.dv2)) / (self.Isp * self.g0)))
        self.mass -= dm_min
        self.mission_done = True
        return 100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.mass = self.initial_mass
        self.I = np.copy(self.base_I)
        self.mission_done = False
        
        self._scripted_hohmann_forward()

        # All bodies on the 700 km orbit: start point 8 m behind the obstacle
        # along-track, debris 8 m ahead; the chaser starts 0.5 m/s faster.
        r0 = self.r_target 
        v0 = np.sqrt(self.mu / r0) 
        
        self.r_ost = np.array([r0, 0.0, 0.0], dtype=np.float64)
        self.v_ost = np.array([0.0, v0, 0.0], dtype=np.float64)
        
        self.r_start = np.array([r0, -8.0, 0.0], dtype=np.float64)
        self.v_start = np.array([0.0, v0, 0.0], dtype=np.float64)
        
        self.r_debris = np.array([r0, 8.0, 0.0], dtype=np.float64)
        self.v_debris = np.array([0.0, v0, 0.0], dtype=np.float64)
        
        self.r_chaser = np.copy(self.r_start)
        self.v_chaser = np.array([0.0, v0 + 0.5, 0.0], dtype=np.float64)
        
        self.r_goal = np.copy(self.r_debris)
        self.v_goal = np.copy(self.v_debris)
        
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) 
        self.omega = np.array([0.0, 0.0, 0.0], dtype=np.float64) 
        
        self.phase = 1.0 # 1.0 = approach and capture, 2.0 = return with the debris
        
        self.ghost_debris_lvlh = None
        self.current_step = 0
        self.max_steps = 1000 
        
        self.traj_out_x, self.traj_out_y, self.traj_out_z = [0.0], [-8.0], [0.0]
        self.traj_back_x, self.traj_back_y, self.traj_back_z = [], [], []
        self.nose_local_x, self.nose_local_y, self.nose_local_z = [0.0], [-6.0], [0.0]
        
        self.prev_dist_goal = np.linalg.norm(self.r_chaser - self.r_goal)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Relative states expressed in the body frame
        rel_pos_target = self.r_goal - self.r_chaser
        rel_vel_target = self.v_goal - self.v_chaser
        rel_pos_ost = self.r_ost - self.r_chaser
        
        R_body_to_ECI = q_to_rot_mat(self.q)
        R_ECI_to_body = R_body_to_ECI.T
        
        target_pos_body = R_ECI_to_body @ rel_pos_target
        target_vel_body = R_ECI_to_body @ rel_vel_target
        ost_pos_body = R_ECI_to_body @ rel_pos_ost
        
        mass_fraction = self.mass / self.initial_mass
        
        obs = np.array([
            target_pos_body[0]/10, target_pos_body[1]/10, target_pos_body[2]/10,
            target_vel_body[0]/5, target_vel_body[1]/5, target_vel_body[2]/5,
            ost_pos_body[0]/10, ost_pos_body[1]/10, ost_pos_body[2]/10,
            self.omega[0]/0.5, self.omega[1]/0.5, self.omega[2]/0.5,
            self.q[0], self.q[1], self.q[2], self.q[3],
            mass_fraction,
            self.phase 
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

    def step(self, action):
        self.current_step += 1
        
        thrust_mapping = {0: -self.max_thrust, 1: 0.0, 2: self.max_thrust}
        torque_mapping = {0: -self.max_torque, 1: 0.0, 2: self.max_torque}
        
        F_body = np.array([thrust_mapping[action[0]], thrust_mapping[action[1]], thrust_mapping[action[2]]])
        T_body = np.array([torque_mapping[action[3]], torque_mapping[action[4]], torque_mapping[action[5]]])
        
        # Torque also comes from thrusters, so it is charged against propellant too
        total_thrust_magnitude = np.sum(np.abs(F_body)) + np.sum(np.abs(T_body))
        dm = (total_thrust_magnitude / (self.Isp * self.g0)) * self.dt
        self.mass -= dm
        
        terminated = False
        reward = 0.0
        
        if self.mass <= self.dry_mass:
            self.mass = self.dry_mass
            reward -= 10.0
            terminated = True
            
        omega_dot = (T_body - np.cross(self.omega, self.I * self.omega)) / self.I
        self.omega += omega_dot * self.dt
        
        self.q += q_dot(self.q, self.omega) * self.dt
        self.q /= np.linalg.norm(self.q) 
        
        R_body_to_ECI = q_to_rot_mat(self.q)
        F_ECI = R_body_to_ECI @ F_body
        
        a_ost = self.get_gravity(self.r_ost)
        self.v_ost += a_ost * self.dt
        self.r_ost += self.v_ost * self.dt
        
        a_chaser = self.get_gravity(self.r_chaser) + (F_ECI / self.mass)
        self.v_chaser += a_chaser * self.dt
        self.r_chaser += self.v_chaser * self.dt
        
        a_start = self.get_gravity(self.r_start)
        self.v_start += a_start * self.dt
        self.r_start += self.v_start * self.dt
        
        a_debris = self.get_gravity(self.r_debris)
        self.v_debris += a_debris * self.dt
        self.r_debris += self.v_debris * self.dt
        
        if self.phase == 1.0:
            self.r_goal = np.copy(self.r_debris)
            self.v_goal = np.copy(self.v_debris)
        else:
            self.r_goal = np.copy(self.r_start)
            self.v_goal = np.copy(self.v_start)
            # Captured debris rides 2 m ahead of the chaser along body +y (the nose)
            self.r_debris = self.r_chaser + (R_body_to_ECI @ np.array([0.0, 2.0, 0.0]))
        
        Q_ECI_to_LVLH = self.get_lvlh_matrix(self.r_ost, self.v_ost).T
        local_pos_chaser = Q_ECI_to_LVLH @ (self.r_chaser - self.r_ost)
        
        nose_body = np.array([0.0, 2.0, 0.0]) 
        nose_ECI = self.r_chaser + (R_body_to_ECI @ nose_body)
        local_nose = Q_ECI_to_LVLH @ (nose_ECI - self.r_ost)
        
        if self.phase == 1.0:
            self.traj_out_x.append(local_pos_chaser[0])
            self.traj_out_y.append(local_pos_chaser[1])
            self.traj_out_z.append(local_pos_chaser[2])
        else:
            self.traj_back_x.append(local_pos_chaser[0])
            self.traj_back_y.append(local_pos_chaser[1])
            self.traj_back_z.append(local_pos_chaser[2])
            
        self.nose_local_x.append(local_nose[0])
        self.nose_local_y.append(local_nose[1])
        self.nose_local_z.append(local_nose[2])
        
        dist_goal = np.linalg.norm(self.r_chaser - self.r_goal)
        dist_ost = np.linalg.norm(self.r_chaser - self.r_ost)
        vel_mag = np.linalg.norm(self.v_chaser - self.v_goal)
        omega_mag = np.linalg.norm(self.omega)
        
        reward -= 0.005 * dist_goal
        progress = self.prev_dist_goal - dist_goal
        reward += progress * 0.1 
        self.prev_dist_goal = dist_goal
        reward -= dm * 0.5 
        
        # Pointing reward: body +y (the capture side) should face the goal
        vec_to_target_body =(R_body_to_ECI.T @ (self.r_goal - self.r_chaser)) / max(dist_goal, 1e-6)
        pointing_acc = vec_to_target_body[1] 
        reward += pointing_acc * 0.02 
        
        if omega_mag > 0.1:
            reward -= omega_mag * 0.05
            
        # Penalty grows exponentially inside twice the obstacle radius
        if dist_ost < self.ost_r * 2.0:
            p =(self.ost_r * 2.0 - dist_ost) / self.ost_r
            reward -= 0.1 * (np.exp(3.0 * p) - 1.0)
        
        truncated = False
        
        if np.abs(local_pos_chaser[0]) > 20 or np.abs(local_pos_chaser[1]) > 20 or np.abs(local_pos_chaser[2]) > 20:
            reward -= 5.0
            terminated = True
        elif dist_ost <= self.ost_r:
            reward -= 5.0 
            terminated = True
        
        # Capture needs a slow, non-tumbling, nose-on arrival; otherwise the episode ends
        if dist_goal <= self.goal_radius and self.phase == 1.0:
            if vel_mag < 0.2 and omega_mag < 0.05 and pointing_acc > 0.9:
                print(f"[{self.current_step}] debris captured, starting return")
                reward += 50.0
                self.phase = 2.0

                # Capture point, kept for the plots
                self.ghost_debris_lvlh = Q_ECI_to_LVLH @ (self.r_debris - self.r_ost)
                
                self.mass += self.debris_mass
                # Debris sits on the body y axis, so it mostly adds inertia about x and z
                self.I +=np.array([400.0, 10.0, 400.0])
                
                self.r_goal = np.copy(self.r_start)
                self.v_goal = np.copy(self.v_start)
                dist_goal = np.linalg.norm(self.r_chaser - self.r_goal)
                self.prev_dist_goal = dist_goal 
                
                self.traj_back_x.append(local_pos_chaser[0])
                self.traj_back_y.append(local_pos_chaser[1])
                self.traj_back_z.append(local_pos_chaser[2])
                
            else:
                reward -= 10.0 
                terminated = True
                
        elif dist_goal <= self.goal_radius and self.phase == 2.0:
            if vel_mag < 0.2 and omega_mag < 0.05: 
                print(f"[{self.current_step}] back at the start point, running Phase 3")
                reward += 100.0

                reward += self._scripted_hohmann_return()
                terminated = True
            else:
                reward += 10.0 
                terminated = True
                
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """Phase 0 and 3 orbit plots (2D and 3D ECI) plus the 3D proximity plot."""
        DARK = '#0d1117'
        GRID = '#21262d'

        vehicle_mass = self.mass - (self.debris_mass if self.phase >= 2.0 else 0.0)
        fuel_used = self.initial_mass - vehicle_mass
        status_text = 'SUCCESS' if self.mission_done else 'IN PROGRESS'
        
        def _draw_orbits(ax, rings):
            θ = np.linspace(0, 2*np.pi, 720)
            ax.add_patch(plt.Circle((0, 0), self.Re/1e6, color='#1c6ea4', alpha=0.55, zorder=0))
            for r_orb, col, lbl in rings:
                ax.plot(r_orb/1e6 * np.cos(θ), r_orb/1e6 * np.sin(θ), '--', color=col, lw=0.8, alpha=0.55, label=lbl)

        lim = (self.r_target + 250_000.) / 1e6

        # Phase 0, 2D
        fig0, ax0 = plt.subplots(figsize=(6, 6), facecolor=DARK)
        ax0.set_facecolor(DARK)
        ax0.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        ax0.set_title(f"Phase 0 — OUTBOUND (Red) | Hydrazine used: {fuel_used:.2f} kg", color='white')
        _draw_orbits(ax0, [(self.r_park, '#3fb950', "Park (600km)"), (self.r_target, '#79c0ff', "Target (700km)")])
        
        t0 = np.array([p for p, ph in zip(self.traj_eci_phase03, self.traj_phase_log_03) if ph == 0]) / 1e6
        if len(t0) > 0:
            ax0.plot(t0[:, 0], t0[:, 1], color='#ff4444', lw=2.0, label='Transfer Orbit (RED)', zorder=3)
            ax0.scatter(t0[-1, 0], t0[-1, 1], color='#e3b341', s=80, marker='^', zorder=4, label='Chaser (Finish)')
        
        ax0.set_xlim(-lim, lim); ax0.set_ylim(-lim, lim)
        ax0.legend(facecolor='#161b22', labelcolor='white')

        # Phases 1-2 in the obstacle's LVLH frame
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        x_ost = self.ost_r * np.cos(u) * np.sin(v)
        y_ost = self.ost_r * np.sin(u) * np.sin(v)
        z_ost = self.ost_r * np.cos(v)
        ax1.plot_wireframe(x_ost, y_ost, z_ost, color="red", alpha=0.3, label="Obstacle")
        
        ax1.scatter(0, -8.0, 0, color='green', s=150, marker='s', label='Base (Start/Return)')

        curr_x = self.traj_out_x[-1] if self.phase == 1.0 else self.traj_back_x[-1]
        curr_y = self.traj_out_y[-1] if self.phase == 1.0 else self.traj_back_y[-1]
        curr_z = self.traj_out_z[-1] if self.phase == 1.0 else self.traj_back_z[-1]
        
        ax1.scatter(curr_x, curr_y, curr_z, color="#f17e11", s=100, marker='^', label='Chaser (Satellite)')

        ax1.plot([curr_x, self.nose_local_x[-1]], 
                 [curr_y, self.nose_local_y[-1]], 
                 [curr_z, self.nose_local_z[-1]], 
                 color='orange', linewidth=4, label='Nose Alignment')
        
        if self.phase == 1.0:
            ax1.scatter(0, 8.0, 0, color='magenta', s=200, marker='*', label='Debris')
        else:
            # After capture: hollow star at the capture point, filled star on the nose
            if self.ghost_debris_lvlh is not None:
                ax1.scatter(self.ghost_debris_lvlh[0], self.ghost_debris_lvlh[1], self.ghost_debris_lvlh[2], 
                            facecolors='none', edgecolors='magenta', s=200, marker='*', linewidths=1.5, alpha=0.5, label='Capture Point')
            
            if len(self.nose_local_x) > 0:
                ax1.scatter(self.nose_local_x[-1], self.nose_local_y[-1], self.nose_local_z[-1], 
                            color='magenta', s=200, marker='*', label='Debris (Trailer)')
        
        ax1.plot(self.traj_out_x, self.traj_out_y, self.traj_out_z, color='blue', linewidth=2, label='Stage 1: Approach')
        if len(self.traj_back_x) > 0:
            ax1.plot(self.traj_back_x, self.traj_back_y, self.traj_back_z, color='purple', linewidth=2, linestyle='--', label='Stage 2: Return (Load)')
        
        ax1.plot([self.traj_out_x[-1] if self.phase==1.0 else self.traj_back_x[-1], self.nose_local_x[-1]], 
                 [self.traj_out_y[-1] if self.phase==1.0 else self.traj_back_y[-1], self.nose_local_y[-1]], 
                 [self.traj_out_z[-1] if self.phase==1.0 else self.traj_back_z[-1], self.nose_local_z[-1]], 
                 color='orange', linewidth=4, label='Nose Alignment')
        
        ax1.set_xlim([-10, 10]); ax1.set_ylim([-12, 12]); ax1.set_zlim([-10, 10])
        ax1.set_xlabel("X (R-Bar) [m]"); ax1.set_ylabel("Y (V-Bar) [m]"); ax1.set_zlabel("Z (H-Bar) [m]")
        ax1.set_title(f"Mission Complete - Status: {status_text} | Total hydrazine: {fuel_used:.2f} kg", color='white', fontsize=11)
        ax1.legend(loc='lower right')

        # Phase 3, 2D
        fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor=DARK)
        ax2.set_facecolor(DARK)
        ax2.set_title(f"Phase 3 — RETURN (Yellow) | Spent hydrazine: {fuel_used:.2f} kg", color='white')
        ax2.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        _draw_orbits(ax2, [(self.r_target, '#79c0ff', "Target (700km)"), (self.r_park, '#3fb950', "Safe zone (600km)")])
        
        t3 = np.array([p for p, ph in zip(self.traj_eci_phase03, self.traj_phase_log_03) if ph == 3]) / 1e6
        if len(t3) > 0:
            ax2.plot(t3[:, 0], t3[:, 1], color='#ffcc00', lw=2.0, label='Return with cargo', zorder=3)
            ax2.scatter(t3[-1, 0], t3[-1, 1], color='#e3b341', s=80, marker='s', zorder=4, label='Chaser at the Base')
        
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
        ax2.legend(facecolor='#161b22', labelcolor='white')

        # 3D ECI views
        vehicle_mass = self.mass - (self.debris_mass if self.phase >= 2.0 else 0.0)
        fuel_used_3d = self.initial_mass - vehicle_mass

        traj0_3d = [p for p, ph in zip(self.traj_eci_phase03, self.traj_phase_log_03) if ph == 0]
        traj3_3d = [p for p, ph in zip(self.traj_eci_phase03, self.traj_phase_log_03) if ph == 3]

        def _style_ax3d_eci(ax, title):
            ax.set_facecolor('#0d1117')
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            ax.tick_params(colors='#8b949e', labelsize=7)
            ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
            ax.set_title(title, color='white', fontsize=10, pad=10)
            ax.set_xlabel("X [×10⁶ m]", color='#8b949e', fontsize=8)
            ax.set_ylabel("Y [×10⁶ m]", color='#8b949e', fontsize=8)
            ax.set_zlabel("Z [×10⁶ m]", color='#8b949e', fontsize=8)

        def _draw_earth_3d(ax):
            u, v = np.mgrid[0:2*np.pi:36j, 0:np.pi:18j]
            x_earth = (self.Re/1e6) * np.cos(u) * np.sin(v)
            y_earth = (self.Re/1e6) * np.sin(u) * np.sin(v)
            z_earth = (self.Re/1e6) * np.cos(v)
            ax.plot_surface(x_earth, y_earth, z_earth, color='#1c6ea4', alpha=0.15, shade=False, zorder=0)
            
            θ = np.linspace(0, 2*np.pi, 360)
            ax.plot((self.r_park/1e6)*np.cos(θ), (self.r_park/1e6)*np.sin(θ), np.zeros_like(θ), '--', color='#3fb950', lw=0.8, alpha=0.55, label="Park (600km)")
            ax.plot((self.r_target/1e6)*np.cos(θ), (self.r_target/1e6)*np.sin(θ), np.zeros_like(θ), '--', color='#79c0ff', lw=0.8, alpha=0.55, label="Target (700km)")

        lim_eci = (self.r_target + 250_000.) / 1e6

        # Phase 0, 3D
        fig_3d_0 = plt.figure(figsize=(8, 8), facecolor='#0d1117')
        ax_3d_0 = fig_3d_0.add_subplot(111, projection='3d')
        _style_ax3d_eci(ax_3d_0, f"Phase 0 — 3D OUTBOUND | Hydrazine: {fuel_used_3d:.2f} kg")
        _draw_earth_3d(ax_3d_0)
        
        if len(traj0_3d) > 1:
            t0_3d = np.array(traj0_3d) / 1e6
            ax_3d_0.plot(t0_3d[:, 0], t0_3d[:, 1], t0_3d[:, 2], color='#ff4444', lw=2.0, label='First leg (Phase 0)', zorder=3)
            ax_3d_0.scatter(t0_3d[-1, 0], t0_3d[-1, 1], t0_3d[-1, 2], color='#e3b341', s=80, marker='^', zorder=4, label='Chaser')
            ax_3d_0.scatter(self.r_debris[0]/1e6, self.r_debris[1]/1e6, self.r_debris[2]/1e6, color='magenta', s=100, marker='*', zorder=4, label='Debris')
        
        ax_3d_0.set_xlim(-lim_eci, lim_eci); ax_3d_0.set_ylim(-lim_eci, lim_eci); ax_3d_0.set_zlim(-lim_eci, lim_eci)
        ax_3d_0.legend(loc='lower left', fontsize=8, facecolor='#161b22', labelcolor='white')

        # Phase 3, 3D
        fig_3d_3 = plt.figure(figsize=(8, 8), facecolor='#0d1117')
        ax_3d_3 = fig_3d_3.add_subplot(111, projection='3d')
        _style_ax3d_eci(ax_3d_3, f"Stage 3 — RETURN 3D | Hydrazine: {fuel_used_3d:.2f} kg")
        _draw_earth_3d(ax_3d_3)
        
        if len(traj3_3d) > 1:
            t3_3d = np.array(traj3_3d) / 1e6
            ax_3d_3.plot(t3_3d[:, 0], t3_3d[:, 1], t3_3d[:, 2], color='#ffcc00', lw=2.0, label='Return with a load', zorder=3)
            ax_3d_3.scatter(t3_3d[-1, 0], t3_3d[-1, 1], t3_3d[-1, 2], color='green', s=100, marker='s', zorder=4, label='Return beacon')
            ax_3d_3.scatter(t3_3d[-1, 0], t3_3d[-1, 1], t3_3d[-1, 2], color='magenta', s=100, marker='*', zorder=5, label='Debris (captured)') 

        ax_3d_3.set_xlim(-lim_eci, lim_eci); ax_3d_3.set_ylim(-lim_eci, lim_eci); ax_3d_3.set_zlim(-lim_eci, lim_eci)
        ax_3d_3.legend(loc='lower left', fontsize=8, facecolor='#161b22', labelcolor='white')

        plt.show()

    def save_gif(self, filename="adr_mission.gif"):
        """Animate the proximity plot (Phases 1-2) to a GIF."""
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        x_ost = self.ost_r * np.cos(u) * np.sin(v)
        y_ost = self.ost_r * np.sin(u) * np.sin(v)
        z_ost = self.ost_r * np.cos(v)
        ax.plot_wireframe(x_ost, y_ost, z_ost, color="red", alpha=0.3, label="Obstacle")
        ax.scatter(0, -8.0, 0, color='green', s=150, marker='s', label='Base (Start/Return)')

        debris_marker, = ax.plot([], [], [], marker='*', color='magenta', markersize=14, ls='', label='Debris (Vespa)')
        ghost_marker, = ax.plot([], [], [], marker='*', markeredgecolor='magenta', markerfacecolor='none', markersize=14, markeredgewidth=1.5, alpha=0.5, ls='', label='Capture Point')

        chaser_marker, = ax.plot([], [], [], marker='^', color="#f17e11", markersize=10, ls='', label='Chaser')

        line_out, = ax.plot([], [], [], color='blue', linewidth=2, label='Stage 1: Approach')
        line_back, = ax.plot([], [], [], color='purple', linewidth=2, linestyle='--', label='Stage 2: Return (Load)')
        nose_line, = ax.plot([], [], [], color='orange', linewidth=4, label='Nose Alignment')

        ax.set_xlim([-10, 10])
        ax.set_ylim([-12, 12])
        ax.set_zlim([-10, 10])
        ax.set_xlabel("X (R-Bar) [m]")
        ax.set_ylabel("Y (V-Bar) [m]")
        ax.set_zlabel("Z (H-Bar) [m]")

        ax.legend(loc='lower right')

        x_out, y_out, z_out = self.traj_out_x, self.traj_out_y, self.traj_out_z
        x_back, y_back, z_back = self.traj_back_x, self.traj_back_y, self.traj_back_z
        nx, ny, nz = self.nose_local_x, self.nose_local_y, self.nose_local_z 
        
        total_steps = len(x_out) + len(x_back)
        if total_steps == 0:
            print("No data to animate.")
            return
            
        step_size = max(1, total_steps // 120) # ~120 frames
        frames_count = total_steps // step_size

        vehicle_mass = self.mass - (self.debris_mass if self.phase >= 2.0 else 0.0)
        fuel_used = self.initial_mass - vehicle_mass
        status_text = 'SUCCESS' if self.mission_done else 'IN PROGRESS'

        def update(frame):
            curr_step = frame * step_size
            g_idx = min(curr_step, len(nx) - 1)
            
            if curr_step < len(x_out):
                # Phase 1
                idx = curr_step
                cx, cy, cz = x_out[idx], y_out[idx], z_out[idx]

                line_out.set_data(x_out[:idx], y_out[:idx])
                line_out.set_3d_properties(z_out[:idx])

                line_back.set_data([], [])
                line_back.set_3d_properties([])
                ghost_marker.set_data([], [])
                ghost_marker.set_3d_properties([])

                debris_marker.set_data([0], [8.0])
                debris_marker.set_3d_properties([0])
                debris_marker.set_label('Debris (Vespa)')

            else:
                # Phase 2: debris follows the nose
                idx = min(curr_step - len(x_out), len(x_back) - 1)
                cx, cy, cz = x_back[idx], y_back[idx], z_back[idx]
                
                line_out.set_data(x_out, y_out)
                line_out.set_3d_properties(z_out)
                
                line_back.set_data(x_back[:idx+1], y_back[:idx+1])
                line_back.set_3d_properties(z_back[:idx+1])

                debris_marker.set_data([nx[g_idx]], [ny[g_idx]])
                debris_marker.set_3d_properties([nz[g_idx]])
                debris_marker.set_label('Debris (Trailer)')

                if self.ghost_debris_lvlh is not None:
                    ghost_marker.set_data([self.ghost_debris_lvlh[0]], [self.ghost_debris_lvlh[1]])
                    ghost_marker.set_3d_properties([self.ghost_debris_lvlh[2]])

            chaser_marker.set_data([cx], [cy])
            chaser_marker.set_3d_properties([cz])

            nose_line.set_data([cx, nx[g_idx]], [cy, ny[g_idx]])
            nose_line.set_3d_properties([cz, nz[g_idx]])

            ax.set_title(f"Mission Complete - Status: {status_text} | Total hydrazine: {fuel_used:.2f} kg", fontsize=11)
                
            return line_out, line_back, debris_marker, ghost_marker, nose_line, chaser_marker
            
        print("\nWriting GIF")
        ani = animation.FuncAnimation(fig, update, frames=frames_count, interval=50, blit=False)
        ani.save(filename, writer='pillow', fps=20)
        print(f"GIF saved as '{filename}'")

        plt.close(fig)

if __name__ == "__main__":
    import os
    import random
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    import scipy.io as sio
    import numpy as np

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    env = ADRFullEnv()
    env.reset(seed=SEED)

    checkpoint_dir = "./adr_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=checkpoint_dir,
        name_prefix="adr_hybrid"
    )

    print("Phases 1-2 training (Phases 0 and 3 scripted)\n")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.995,
        n_steps=8192,
        batch_size=512,
        n_epochs=10,
        ent_coef=0.01,
        clip_range=0.15,
        gae_lambda=0.97,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=SEED,
    )

    # Checkpoints are written every 1M steps, so training can be stopped with Ctrl+C
    # (the test and export below then do not run).
    TOTAL_TIMESTEPS = 5_000_000

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )

    print("\nTraining done, test episode")
    obs, _ = env.reset(seed=SEED)
    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            status = "SUCCESS" if env.mission_done else "FAILED"
            print(f"[{status}] - Step {i+1} - Phase {env.phase}")
            break

    matlab_data = {
        'traj_out_x': np.array(env.traj_out_x),
        'traj_out_y': np.array(env.traj_out_y),
        'traj_out_z': np.array(env.traj_out_z),
        'traj_back_x': np.array(env.traj_back_x),
        'traj_back_y': np.array(env.traj_back_y),
        'traj_back_z': np.array(env.traj_back_z),
        'nose_x': np.array(env.nose_local_x),
        'nose_y': np.array(env.nose_local_y),
        'nose_z': np.array(env.nose_local_z),
        'ost_r': env.ost_r,
        'start_pos': np.array([0, -8.0, 0]),
        'debris_pos': np.array([0, 8.0, 0])
    }
    
    # Must match the file name loaded in plot12.m
    mat_name = "phase1_2_orbits.mat"
    sio.savemat(mat_name, matlab_data)
    print(f"Data exported for MATLAB to '{mat_name}'")
    env.render()
    #env.save_gif("space_mission.gif")
    env.close()

# Alternative entry point: resume training from the latest checkpoint in ./adr_checkpoints/.
'''if __name__ == "__main__":
    import os
    import glob
    import random
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    env = ADRFullEnv()
    env.reset(seed=SEED)

    checkpoint_dir = "./adr_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=checkpoint_dir,
        name_prefix="adr_hybrid"
    )

    print("Phases 1-2: resuming training\n")

    list_of_files = glob.glob(f"{checkpoint_dir}/*.zip")

    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
        print(f"Resuming from {latest_checkpoint}")

        model = PPO.load(latest_checkpoint, env=env)

        TOTAL_TIMESTEPS = 1_000
        print("\nTraining")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=False  # keep the step counter and logs continuous
        )

        print("\nTraining done, test episode")
        obs, _ = env.reset(seed=SEED)
        for i in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                status = "SUCCESS" if env.mission_done else "FAILED"
                print(f"[{status}] - Step {i+1} - Phase {env.phase}")
                break

        env.render()
        env.save_gif("space_mission.gif")

    else:
        print(f"No checkpoint found in {checkpoint_dir}")
        
    env.close()'''
