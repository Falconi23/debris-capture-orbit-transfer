% Phase 3: return overview, altitude profile and arrival close-up.
clear; clc; close all;

%% Load
% Must match mat_name in Phase3.py (one file per curriculum step)
filename = 'phase3_curriculum_step500.mat';
if ~isfile(filename)
    error('File %s not found. Please ensure you are in the correct directory.', filename);
end
load(filename);

theta = linspace(0, 2*pi, 360);

%% Overview
fig1 = figure('Name', 'Phase 3 - Return Journey', 'Color', 'w', 'Position', [100, 200, 1200, 550]);

% Left: ECI X-Y
subplot(1, 2, 1);
hold on; grid on; axis equal;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');

earth_x = Re * cos(theta); earth_y = Re * sin(theta);
fill(earth_x, earth_y, [0.11, 0.43, 0.64], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'Earth');

plot(r_target * cos(theta), r_target * sin(theta), '--', 'Color', [0.47, 0.75, 1.00], 'LineWidth', 1.5, 'DisplayName', 'Departure Orbit (610 km)');
plot(r_park * cos(theta), r_park * sin(theta), '--', 'Color', [0.25, 0.72, 0.31], 'LineWidth', 1.5, 'DisplayName', 'Safe Zone (600 km)');

plot(traj_x, traj_y, 'Color', [0.93, 0.69, 0.13], 'LineWidth', 2.5, 'DisplayName', 'Return Trajectory (with Load)');

plot(traj_x(1), traj_y(1), 'p', 'MarkerSize', 14, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Capture Point (Start)');
plot(traj_x(end), traj_y(end), 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.93, 0.69, 0.13], 'MarkerEdgeColor', 'k', 'DisplayName', 'Spacecraft (End)');

xlabel('X Position [m]', 'FontWeight', 'bold');
ylabel('Y Position [m]', 'FontWeight', 'bold');
title('2D Cartesian View', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 9);

% Whole Earth and both orbits, 10% outside the 610 km orbit
lim = r_target * 1.10;
xlim([-lim, lim]);
ylim([-lim, lim]);

% Right: altitude [km] vs step
subplot(1, 2, 2);
hold on; grid on;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);

alt_traj = (sqrt(traj_x.^2 + traj_y.^2) - Re) / 1e3;
alt_park = (r_park - Re) / 1e3;
alt_target = (r_target - Re) / 1e3;
steps = 1:length(alt_traj);

plot([1, length(steps)], [alt_target, alt_target], '--', 'Color', [0.47, 0.75, 1.00], 'LineWidth', 1.5, 'DisplayName', 'Departure (610 km)');
plot([1, length(steps)], [alt_park, alt_park], '--', 'Color', [0.25, 0.72, 0.31], 'LineWidth', 1.5, 'DisplayName', 'Safe Zone (600 km)');

plot(steps, alt_traj, 'Color', [0.93, 0.69, 0.13], 'LineWidth', 2.5, 'DisplayName', 'Spacecraft Altitude');

plot(1, alt_traj(1), 'p', 'MarkerSize', 14, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Capture Point (Start)');
plot(length(steps), alt_traj(end), 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.93, 0.69, 0.13], 'MarkerEdgeColor', 'k', 'DisplayName', 'Spacecraft (End)');

xlabel('Time (Simulation Steps)', 'FontWeight', 'bold');
ylabel('Altitude [km]', 'FontWeight', 'bold');
title('Altitude Profile (Descent to Safe Zone)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 9); % keeps the legend off the descent

% 5% margin so the end markers are not clipped
x_margin = length(steps) * 0.05;
xlim([1 - x_margin, length(steps) + x_margin]);
ylim([595, 615]);

%% Arrival close-up
fig2 = figure('Name', 'Phase 3 - Arrival Zoom', 'Color', 'w', 'Position', [350, 250, 700, 700]);
hold on; grid on; axis equal;
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);

plot(r_park * cos(theta), r_park * sin(theta), '--', 'Color', [0.25, 0.72, 0.31], 'LineWidth', 2, 'DisplayName', 'Safe Zone (600 km)');

plot(traj_x, traj_y, 'Color', [0.93, 0.69, 0.13], 'LineWidth', 2.5, 'DisplayName', 'Return Trajectory');

plot(traj_x(end), traj_y(end), 's', 'MarkerSize', 14, 'MarkerFaceColor', [0.93, 0.69, 0.13], 'MarkerEdgeColor', 'k', 'DisplayName', 'Spacecraft (End)');

xlabel('X Position [m]', 'FontWeight', 'bold');
ylabel('Y Position [m]', 'FontWeight', 'bold');
title('Terminal Phase: Safe Zone Arrival Precision', 'FontSize', 14);

% ±5 km around the final position
zoom_m = 5000; 
xlim([traj_x(end) - zoom_m, traj_x(end) + zoom_m]);
ylim([traj_y(end) - zoom_m, traj_y(end) + zoom_m]);

legend('Location', 'northeast', 'FontSize', 10);

