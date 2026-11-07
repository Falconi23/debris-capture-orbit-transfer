% Phase 0: transfer overview, altitude profile and rendezvous close-up.
clear; clc; close all;

%% Load
% Must match mat_name in Phase0.py
filename = 'phase0_orbits2500.mat';
if ~isfile(filename)
    error('File %s not found. Please ensure you are in the correct directory.', filename);
end
load(filename);

theta = linspace(0, 2*pi, 360);

%% Overview
fig1 = figure('Name', 'Phase 0 - Orbital Transfer', 'Color', 'w', 'Position', [100, 200, 1200, 550]);

% Left: ECI X-Y
subplot(1, 2, 1);
hold on; grid on; axis equal;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');

earth_x = Re * cos(theta); earth_y = Re * sin(theta);
fill(earth_x, earth_y, [0.11, 0.43, 0.64], 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'Earth');

plot(r_park * cos(theta), r_park * sin(theta), '--', 'Color', [0.25, 0.72, 0.31], 'LineWidth', 1.5, 'DisplayName', 'Park Orbit');
plot(r_target * cos(theta), r_target * sin(theta), '--', 'Color', [0.47, 0.75, 1.00], 'LineWidth', 1.5, 'DisplayName', 'Target Orbit');

plot(traj_x, traj_y, 'Color', 'r', 'LineWidth', 1.5, 'DisplayName', 'Transfer');
plot(traj_x(1), traj_y(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'none', 'DisplayName', 'Start');
plot(traj_x(end), traj_y(end), 's', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Chaser End');
plot(r_debris_final(1), r_debris_final(2), 'p', 'MarkerSize', 12, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Debris');

xlabel('X Position [m]', 'FontWeight', 'bold');
ylabel('Y Position [m]', 'FontWeight', 'bold');
title('2D Cartesian View', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 9);

% Frame the trajectory with a 150 km margin
zoom_margin = 150e3;
xlim([min(traj_x) - zoom_margin, max(traj_x) + zoom_margin]);
ylim([min(traj_y) - zoom_margin, max(traj_y) + zoom_margin]);

% Right: altitude [km] vs step
subplot(1, 2, 2);
hold on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);

alt_traj = (sqrt(traj_x.^2 + traj_y.^2) - Re) / 1e3;
alt_park = (r_park - Re) / 1e3;
alt_target = (r_target - Re) / 1e3;
steps = 1:length(alt_traj);

plot([1, length(steps)], [alt_park, alt_park], '--', 'Color', [0.25, 0.72, 0.31], 'LineWidth', 1.5, 'DisplayName', 'Park (600 km)');
plot([1, length(steps)], [alt_target, alt_target], '--', 'Color', [0.47, 0.75, 1.00], 'LineWidth', 1.5, 'DisplayName', 'Target (610 km)');

plot(steps, alt_traj, 'Color', 'r', 'LineWidth', 1.5, 'DisplayName', 'Chaser Altitude');

plot(1, alt_traj(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'none', 'DisplayName', 'Start');
plot(length(steps), alt_traj(end), 's', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Chaser End');
plot(length(steps), alt_target, 'p', 'MarkerSize', 12, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Debris');

xlabel('Time (Simulation Steps)', 'FontWeight', 'bold');
ylabel('Altitude [km]', 'FontWeight', 'bold');
title('Altitude Profile', 'FontSize', 12);
legend('Location', 'southeast', 'FontSize', 9);

% 5% margin so the end markers are not clipped
x_margin = length(steps) * 0.05;
xlim([1 - x_margin, length(steps) + x_margin]);
ylim([595, 615]);

%% Rendezvous close-up
fig2 = figure('Name', 'Phase 0 - Rendezvous Zoom', 'Color', 'w', 'Position', [350, 250, 700, 700]);
hold on; grid on; axis equal;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);

plot(r_target * cos(theta), r_target * sin(theta), '--', 'Color', [0.47, 0.75, 1.00], 'LineWidth', 2, 'DisplayName', 'Target Orbit');

plot(traj_x, traj_y, 'Color', 'r', 'LineWidth', 1.5, 'DisplayName', 'Chaser Trajectory');

plot(traj_x(end), traj_y(end), 's', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Chaser End');
plot(r_debris_final(1), r_debris_final(2), 'p', 'MarkerSize', 18, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Debris');

xlabel('X Position [m]', 'FontWeight', 'bold');
ylabel('Y Position [m]', 'FontWeight', 'bold');
title('Terminal Phase: Rendezvous Precision', 'FontSize', 14);

% ±5 km around the debris
zoom_m = 5000; 
xlim([r_debris_final(1) - zoom_m, r_debris_final(1) + zoom_m]);
ylim([r_debris_final(2) - zoom_m, r_debris_final(2) + zoom_m]);

legend('Location', 'northeast', 'FontSize', 10);
