% Phases 1-2: 6-DOF proximity operations in the obstacle's LVLH frame.
% Reads the .mat written by Phase12.py.
clear; clc; close all;

load('phase1_2_orbits.mat');

figure('Name', 'Phases 1 and 2 - CPO', 'Color', 'w', 'Position', [100, 100, 900, 700]);
hold on; grid on; axis equal; view(3);

[x_sph, y_sph, z_sph] = sphere(40);
surf(x_sph * ost_r, y_sph * ost_r, z_sph * ost_r, ...
    'FaceColor', 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'Keep-Out Zone (Obstacle)');

plot3(start_pos(1), start_pos(2), start_pos(3), 's', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'DisplayName', 'Start / Return Base');
plot3(debris_pos(1), debris_pos(2), debris_pos(3), 'p', 'MarkerSize', 18, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Space Debris (Target)');

if ~isempty(traj_out_x) && length(traj_out_x) > 1
    plot3(traj_out_x, traj_out_y, traj_out_z, 'b', 'LineWidth', 2.5, 'DisplayName', 'Phase 1: Approach');
end

if ~isempty(traj_back_x) && length(traj_back_x) > 1
    plot3(traj_back_x, traj_back_y, traj_back_z, '--', 'Color', [0.5 0 0.5], 'LineWidth', 2.5, 'DisplayName', 'Phase 2: Return with load');
end

% Final position: end of the return leg if there is one, otherwise end of the approach
end_x = NaN; end_y = NaN; end_z = NaN;
if ~isempty(traj_back_x) && length(traj_back_x) > 1
    end_x = traj_back_x(end); end_y = traj_back_y(end); end_z = traj_back_z(end);
elseif ~isempty(traj_out_x) && length(traj_out_x) > 1
    end_x = traj_out_x(end); end_y = traj_out_y(end); end_z = traj_out_z(end);
end

if ~isnan(end_x) && ~isempty(nose_x)
    plot3([end_x, nose_x(end)], [end_y, nose_y(end)], [end_z, nose_z(end)], 'Color', [1 0.5 0], 'LineWidth', 4, 'DisplayName', 'Spacecraft Attitude (Nose)');
    plot3(end_x, end_y, end_z, '^', 'MarkerSize', 10, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

xlabel('X (R-Bar) [m]', 'FontWeight', 'bold');
ylabel('Y (V-Bar) [m]', 'FontWeight', 'bold');
zlabel('Z (H-Bar) [m]', 'FontWeight', 'bold');

title('Close Proximity Operations (6-DOF)', 'FontSize', 14);

% Same box as the Python plot (the env terminates beyond ±20 m)
xlim([-10 10]);
ylim([-12 12]);
zlim([-10 10]);

legend('Location', 'northeastoutside', 'FontSize', 10);
hold off;
