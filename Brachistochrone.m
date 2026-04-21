% Brachistochrone Gradient Calculations
% Reference: Jameson & Vassberg (2000)

clc
clear all
close all

%% Analytical solution of the Brachistochrone problem

% Parameters from the paper
C = 1;
t_0 = pi/2;
t_1 = pi;

% Generating the Analytical Solution (Exact Cycloid)
t_fine = linspace(t_0, t_1, 100);
x_exact = 0.5 * C^2 * (t_fine - sin(t_fine));   %Equation 10
y_exact = 0.5 * C^2 * (1 - cos(t_fine));        %Equation 9        

% Normalizing x_exact to start at 0 (consistent with mesh setup x = 0 to L)
x_offset = x_exact(1);
% x_offset = 0;
x_exact = x_exact - x_offset;

% Plot: Debugging 
figure('Color', 'w');
plot(x_exact, y_exact, 'r-', 'LineWidth', 2); hold on;
set(gca, 'YDir', 'reverse'); 
xlabel('X');
ylabel('Y');
title('Brachistochrone Solution: Analytical Cycloid');
legend('Analytical Solution (Cycloid)', 'Location', 'best');
grid on;
axis equal;



%% Problem Setup
N = 31;                 % Number of design variables (internal points): tune to 31, 511
L = 1.0;                % Total horizontal distance
dx = L / (N + 1);       % Mesh interval (delta x) 
x = (0:dx:L)';          % Discrete spatial variable x 

% Boundary conditions (y0 and yN+1 are fixed) 
y = linspace(0.5, 1.0, N+2)'; % Initial guess


%% Continuous Gradient Calculation

% Approximates the analytical gradient using 2nd-order finite differences
G_cont = zeros(N, 1);

for j = 2:N+1
    % Finite difference approximations for y' and y''
    dy_dx = (y(j+1) - y(j-1)) / (2 * dx);
    d2y_dx2 = (y(j+1) - 2*y(j) + y(j-1)) / (dx^2);
    
    % Continuous Gradient Formula (Equation 11) 
    numerator = 1 + dy_dx^2 + 2 * y(j) * d2y_dx2;
    denominator = 2 * (y(j) * (1 + dy_dx^2))^1.5;
    G_cont(j-1) = -numerator / denominator;
end

%% Discrete Gradient Calculation 

% Exactly differentiates the discrete cost function (Rectangle Rule) 
G_disc = zeros(N, 1);

% Helper values at half-mesh points (j + 1/2) 
y_mid = 0.5 * (y(2:end) + y(1:end-1));          % y at j+1/2
dy_mid = (y(2:end) - y(1:end-1)) / dx;          % y' at j+1/2

% Calculating A and B components at j+1/2 
A = sqrt(1 + dy_mid.^2) ./ (2 * y_mid.^1.5);
B = dy_mid ./ sqrt(y_mid .* (1 + dy_mid.^2));

% Discrete Gradient Formula (Equation 12) 
for j = 1:N
    % j in this loop corresponds to the j-th design variable (y(j+1))
    % Formula: G_j = B_{j-1/2} - B_{j+1/2} - (dx/2) * (A_{j+1/2} + A_{j-1/2})
    G_disc(j) = B(j) - B(j+1) - (dx/2) * (A(j+1) + A(j));
end

%% Displaying Results
fprintf('L2 Norm of Continuous Gradient: %.6e\n', norm(G_cont));
fprintf('L2 Norm of Discrete Gradient: %.6e\n', norm(G_disc));


% Plotting
figure('Color', 'w');
plot(x_exact, y_exact, 'r-', 'LineWidth', 2); hold on;
hold on
plot(x, y, 'bo--', 'MarkerSize', 4); 
set(gca, 'YDir', 'reverse'); 
xlabel('X');
ylabel('Y');
legend('Analytical Solution (Cycloid)', 'Gradient Solution','Location', 'best');
grid on;
axis equal;