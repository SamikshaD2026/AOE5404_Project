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

%% Optimization: Steepest Descent Using Continuous Gradient
% The steepest descent method iteratively updates the trajectory by stepping
% in the direction opposite to the gradient (Equation 13).
% The update rule is: y^(n+1) = y^n - dt * G^n
% The time step dt is chosen based on a CFL-like stability condition: dt <= dx^2 / (2*beta)

maxIter = 50000;        % Maximum number of iterations before stopping
tol = 1e-6;             % Convergence tolerance on the L2 norm of the gradient
dt = 0.1 * dx^2;        % Stable time step size derived from stability analysis

y_sd_cont = y;                              % Initialize trajectory from the initial guess
cost_hist_cont = zeros(maxIter,1);          % Preallocate cost function history
grad_hist_cont = zeros(maxIter,1);          % Preallocate gradient norm history

for iter = 1:maxIter
    % Compute the continuous gradient at the current trajectory
    G = continuousGradient(y_sd_cont, dx);
    
    % Record cost function and gradient norm for convergence monitoring
    cost_hist_cont(iter) = costFunction(y_sd_cont, dx);
    grad_hist_cont(iter) = norm(G, 2);
    
    % Check for convergence: stop if gradient norm drops below tolerance
    if grad_hist_cont(iter) < tol
        fprintf('Continuous steepest descent converged at iteration %d\n', iter);
        break;
    end
    
    % Update only internal design variables (boundary points y(1) and y(end) are fixed)
    y_sd_cont(2:end-1) = y_sd_cont(2:end-1) - dt * G;
end

% Trim pre-allocated arrays to the actual number of iterations performed
cost_hist_cont = cost_hist_cont(1:iter);
grad_hist_cont = grad_hist_cont(1:iter);

%% Optimization: Steepest Descent Using Discrete Gradient
% Same steepest descent scheme as above, but the gradient is computed by
% exactly differentiating the discrete (rectangle-rule) cost function.
% This avoids the inconsistency between the continuous gradient and the
% discretized cost, which can improve convergence behavior.

y_sd_disc = y;                              % Initialize trajectory from the initial guess
cost_hist_disc = zeros(maxIter,1);          % Preallocate cost function history
grad_hist_disc = zeros(maxIter,1);          % Preallocate gradient norm history

for iter2 = 1:maxIter
    % Compute the discrete gradient at the current trajectory
    G = discreteGradient(y_sd_disc, dx);
    
    % Record cost function and gradient norm for convergence monitoring
    cost_hist_disc(iter2) = costFunction(y_sd_disc, dx);
    grad_hist_disc(iter2) = norm(G, 2);
    
    % Check for convergence: stop if gradient norm drops below tolerance
    if grad_hist_disc(iter2) < tol
        fprintf('Discrete steepest descent converged at iteration %d\n', iter2);
        break;
    end
    
    % Update only internal design variables (boundary points are fixed)
    y_sd_disc(2:end-1) = y_sd_disc(2:end-1) - dt * G;
end

% Trim pre-allocated arrays to the actual number of iterations performed
cost_hist_disc = cost_hist_disc(1:iter2);
grad_hist_disc = grad_hist_disc(1:iter2);

%% Optimization: BFGS Using Continuous Gradient
% The BFGS quasi-Newton method builds an approximation H of the inverse Hessian
% iteratively using the rank-2 update formula from the progress report.
% The search direction at each step is p = -H * G, which approaches the Newton
% direction as H converges to the true inverse Hessian.
% A backtracking line search (Armijo condition) is used to find a step size
% that guarantees sufficient decrease in the cost function.

maxIter_bfgs = 5000;    % BFGS typically converges in far fewer iterations than steepest descent

y_bfgs_cont = y;                                    % Initialize from the same initial guess
G_bfgs_cont = continuousGradient(y_bfgs_cont, dx); % Initial gradient
H_cont = eye(N);                                    % Initialize inverse Hessian as identity

cost_hist_bfgs_cont = zeros(maxIter_bfgs, 1);
grad_hist_bfgs_cont = zeros(maxIter_bfgs, 1);

for iter3 = 1:maxIter_bfgs
    cost_hist_bfgs_cont(iter3) = costFunction(y_bfgs_cont, dx);
    grad_hist_bfgs_cont(iter3) = norm(G_bfgs_cont, 2);

    % Check for convergence
    if grad_hist_bfgs_cont(iter3) < tol
        fprintf('BFGS (continuous gradient) converged at iteration %d\n', iter3);
        break;
    end

    % Compute search direction: quasi-Newton step p = -H * G
    p = -H_cont * G_bfgs_cont;

    % Backtracking line search (Armijo sufficient decrease condition)
    % Ensures cost decreases by at least a fraction c1 of the expected decrease
    alpha = 1.0;        % Start with full Newton step
    c1    = 1e-4;       % Armijo parameter
    rho   = 0.5;        % Step reduction factor per backtrack
    I0    = cost_hist_bfgs_cont(iter3);

    for ls = 1:30
        y_trial = y_bfgs_cont;
        y_trial(2:end-1) = y_trial(2:end-1) + alpha * p;
        % Only accept step if all y values remain positive (physically required)
        if all(y_trial(2:end-1) > 0) && costFunction(y_trial, dx) <= I0 + c1 * alpha * (G_bfgs_cont' * p)
            break;
        end
        alpha = rho * alpha;
    end

    % Accept the step and compute new gradient
    y_new = y_bfgs_cont;
    y_new(2:end-1) = y_new(2:end-1) + alpha * p;
    G_new = continuousGradient(y_new, dx);

    % BFGS update vectors (changes in design variables and gradient)
    delta_y = y_new(2:end-1) - y_bfgs_cont(2:end-1);  % s in standard BFGS notation
    delta_g = G_new - G_bfgs_cont;                     % y in standard BFGS notation
    curvature = delta_g' * delta_y;

    % Apply BFGS rank-2 inverse Hessian update only if curvature condition holds
    % (delta_g' * delta_y > 0 ensures H remains positive definite)
    if curvature > 1e-10
        H_cont = H_cont ...
            + (1 + (delta_g' * H_cont * delta_g) / curvature) * (delta_y * delta_y') / curvature ...
            - (H_cont * delta_g * delta_y' + delta_y * delta_g' * H_cont) / curvature;
    end

    y_bfgs_cont   = y_new;
    G_bfgs_cont   = G_new;
end

cost_hist_bfgs_cont = cost_hist_bfgs_cont(1:iter3);
grad_hist_bfgs_cont = grad_hist_bfgs_cont(1:iter3);

%% Optimization: BFGS Using Discrete Gradient
% Same BFGS scheme as above, but uses the discrete gradient (exactly consistent
% with the rectangle-rule cost function) to drive the inverse Hessian updates.

y_bfgs_disc = y;                                    % Initialize from the same initial guess
G_bfgs_disc = discreteGradient(y_bfgs_disc, dx);   % Initial gradient
H_disc = eye(N);                                    % Initialize inverse Hessian as identity

cost_hist_bfgs_disc = zeros(maxIter_bfgs, 1);
grad_hist_bfgs_disc = zeros(maxIter_bfgs, 1);

for iter4 = 1:maxIter_bfgs
    cost_hist_bfgs_disc(iter4) = costFunction(y_bfgs_disc, dx);
    grad_hist_bfgs_disc(iter4) = norm(G_bfgs_disc, 2);

    % Check for convergence
    if grad_hist_bfgs_disc(iter4) < tol
        fprintf('BFGS (discrete gradient) converged at iteration %d\n', iter4);
        break;
    end

    % Compute search direction: quasi-Newton step p = -H * G
    p = -H_disc * G_bfgs_disc;

    % Backtracking line search (Armijo sufficient decrease condition)
    alpha = 1.0;
    c1    = 1e-4;
    rho   = 0.5;
    I0    = cost_hist_bfgs_disc(iter4);

    for ls = 1:30
        y_trial = y_bfgs_disc;
        y_trial(2:end-1) = y_trial(2:end-1) + alpha * p;
        if all(y_trial(2:end-1) > 0) && costFunction(y_trial, dx) <= I0 + c1 * alpha * (G_bfgs_disc' * p)
            break;
        end
        alpha = rho * alpha;
    end

    % Accept the step and compute new gradient
    y_new = y_bfgs_disc;
    y_new(2:end-1) = y_new(2:end-1) + alpha * p;
    G_new = discreteGradient(y_new, dx);

    % BFGS update vectors
    delta_y = y_new(2:end-1) - y_bfgs_disc(2:end-1);
    delta_g = G_new - G_bfgs_disc;
    curvature = delta_g' * delta_y;

    % Apply BFGS rank-2 inverse Hessian update only if curvature condition holds
    if curvature > 1e-10
        H_disc = H_disc ...
            + (1 + (delta_g' * H_disc * delta_g) / curvature) * (delta_y * delta_y') / curvature ...
            - (H_disc * delta_g * delta_y' + delta_y * delta_g' * H_disc) / curvature;
    end

    y_bfgs_disc  = y_new;
    G_bfgs_disc  = G_new;
end

cost_hist_bfgs_disc = cost_hist_bfgs_disc(1:iter4);
grad_hist_bfgs_disc = grad_hist_bfgs_disc(1:iter4);

%% Compare Optimized Paths Against Analytical Cycloid
% Interpolate the analytical cycloid onto the uniform mesh x to allow
% a point-wise comparison with the numerically optimized trajectories.
% The L2 error is normalized by sqrt(N+2) to make it mesh-independent.

y_exact_interp = interp1(x_exact, y_exact, x, 'linear', 'extrap');

% Normalized L2 error between each optimized path and the analytical solution
L2_error_cont      = norm(y_sd_cont    - y_exact_interp, 2) / sqrt(length(y));
L2_error_disc      = norm(y_sd_disc    - y_exact_interp, 2) / sqrt(length(y));
L2_error_bfgs_cont = norm(y_bfgs_cont  - y_exact_interp, 2) / sqrt(length(y));
L2_error_bfgs_disc = norm(y_bfgs_disc  - y_exact_interp, 2) / sqrt(length(y));

fprintf('L2 Error Steepest Descent - Continuous Gradient: %.6e\n', L2_error_cont);
fprintf('L2 Error Steepest Descent - Discrete Gradient:   %.6e\n', L2_error_disc);
fprintf('L2 Error BFGS - Continuous Gradient:             %.6e\n', L2_error_bfgs_cont);
fprintf('L2 Error BFGS - Discrete Gradient:               %.6e\n', L2_error_bfgs_disc);

% Plot all paths together for visual comparison
figure('Color','w');
plot(x_exact, y_exact, 'r-', 'LineWidth', 2); hold on;
plot(x, y, 'k--', 'LineWidth', 1.5);
plot(x, y_sd_cont, 'bo-', 'MarkerSize', 4);
plot(x, y_sd_disc, 'gs-', 'MarkerSize', 4);
plot(x, y_bfgs_cont, 'b^--', 'MarkerSize', 4);
plot(x, y_bfgs_disc, 'gd--', 'MarkerSize', 4);
set(gca, 'YDir', 'reverse');   % Flip y-axis so descent is downward visually
xlabel('X');
ylabel('Y');
title('Optimized Brachistochrone Paths');
legend('Analytical Cycloid', 'Initial Guess', ...
       'Steepest Descent - Continuous', 'Steepest Descent - Discrete', ...
       'BFGS - Continuous', 'BFGS - Discrete', ...
       'Location', 'best');
grid on;
axis equal;

%% Plot Convergence Histories
% Semi-log plots allow visualization of the exponential decay in the
% gradient norm and cost function as the optimizer converges.
% BFGS is expected to converge in significantly fewer iterations than
% steepest descent due to its superlinear convergence rate.

% Gradient norm convergence: indicates how close the trajectory is to the optimum
figure('Color','w');
semilogy(grad_hist_cont, 'b-', 'LineWidth', 2); hold on;
semilogy(grad_hist_disc, 'g-', 'LineWidth', 2);
semilogy(grad_hist_bfgs_cont, 'b--', 'LineWidth', 2);
semilogy(grad_hist_bfgs_disc, 'g--', 'LineWidth', 2);
xlabel('Iteration');
ylabel('L2 Norm of Gradient');
title('Gradient Convergence History');
legend('SD - Continuous', 'SD - Discrete', 'BFGS - Continuous', 'BFGS - Discrete', 'Location', 'best');
grid on;

% Cost function convergence: shows reduction in total travel time over iterations
figure('Color','w');
semilogy(cost_hist_cont, 'b-', 'LineWidth', 2); hold on;
semilogy(cost_hist_disc, 'g-', 'LineWidth', 2);
semilogy(cost_hist_bfgs_cont, 'b--', 'LineWidth', 2);
semilogy(cost_hist_bfgs_disc, 'g--', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Cost Function I');
title('Cost Function Convergence History');
legend('SD - Continuous', 'SD - Discrete', 'BFGS - Continuous', 'BFGS - Discrete', 'Location', 'best');
grid on;



%% Helper Functions

% Computes the discrete cost function I using the rectangle (midpoint) rule.
% y_mid and dy_mid are evaluated at the half-mesh points j+1/2.
function I = costFunction(y, dx)
    y_mid = 0.5 * (y(2:end) + y(1:end-1));      % y at j+1/2
    dy_mid = (y(2:end) - y(1:end-1)) / dx;      % y' at j+1/2

    % Integrand F evaluated at each half-mesh point
    F_mid = sqrt(1 + dy_mid.^2) ./ sqrt(y_mid);
    I = sum(F_mid) * dx;    % Rectangle rule integration
end

% Computes the continuous gradient G using 2nd-order finite differences
% to approximate y' and y'' in the analytical gradient formula (Equation 11).
% Boundary points are excluded; only the N internal design variables are returned.
function G_cont = continuousGradient(y, dx)
    N = length(y) - 2;
    G_cont = zeros(N, 1);

    for j = 2:N+1
        % 2nd-order central differences for first and second derivatives
        dy_dx = (y(j+1) - y(j-1)) / (2 * dx);
        d2y_dx2 = (y(j+1) - 2*y(j) + y(j-1)) / dx^2;

        % Continuous gradient formula (Equation 11)
        numerator = 1 + dy_dx^2 + 2 * y(j) * d2y_dx2; 
        denominator = 2 * (y(j) * (1 + dy_dx^2))^(3/2);

        G_cont(j-1) = -numerator / denominator;
    end
end

% Computes the discrete gradient G by analytically differentiating the
% rectangle-rule cost function I_R with respect to each design variable y_j.
% This ensures the gradient is exactly consistent with the discrete cost,
% which avoids truncation errors that can arise with the continuous gradient.
function G_disc = discreteGradient(y, dx)
    N = length(y) - 2;
    G_disc = zeros(N, 1);

    % Evaluate intermediate quantities at half-mesh points j+1/2
    y_mid = 0.5 * (y(2:end) + y(1:end-1));
    dy_mid = (y(2:end) - y(1:end-1)) / dx;

    % A and B are the two components that arise from differentiating I_R
    A = sqrt(1 + dy_mid.^2) ./ (2 * y_mid.^(3/2));     % Contribution from y_mid term
    B = dy_mid ./ sqrt(y_mid .* (1 + dy_mid.^2));       % Contribution from dy_mid term

    % Discrete gradient formula (Equation 12): G_j = dI_R / dy_j
    % Each design variable y_j appears in intervals [j-1/2] and [j+1/2]
    for j = 1:N
        G_disc(j) = B(j) - B(j+1) - (dx/2) * (A(j+1) + A(j));
    end
end