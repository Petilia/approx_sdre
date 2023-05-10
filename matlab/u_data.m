function res = u_data(plot_flag, velocity0, angle0, u_func, r_earth, I, T, Q, R)
    % w_0 = circular orbital rate = g / R^3, I = moment of inertia
    g = 9.807;
    if nargin < 3
        plot_flag = 0;
        angle0 = [77.1834, 16.2300, 90.2594, 3.0]';
        velocity0 = [0.1301, 0.1637, 0.1174]';
        angle0 = deg2rad(angle0);
    end
    if nargin < 4
        u_func = @sdre;
    end
    if nargin < 8
        r_earth = 600;
        I = 10 * eye(3);
        T = 100;
        Q = 0.5 * eye(6);
        R = eye(3);
    end
    I_inv = inv(I);
    w_0 = g / r_earth^3;

    [t_plot, X] = ode45(@(t, x) spacecraft_right_hand_sides(t, x, u_func, w_0, I, I_inv, Q, R), ...
                        0:1:T, [velocity0; angle0]);
    if plot_flag
        % просто выводим графики
        figure()
        plot(t_plot, X(:, 1), t_plot, X(:, 4))
        legend('velocity1', 'angle1')
        figure()
        plot(t_plot, X(:, 2), t_plot, X(:, 5))
        legend('velocity2', 'angle2')
        figure()
        plot(t_plot, X(:, 3), t_plot, X(:, 6))
        legend('velocity3', 'angle3')
        figure()
        plot(t_plot, X(:, 7))
        legend('angle4')
        res = X;
    else
        % иначе - сохраняем данные в формате (x, u)
        examples = zeros(length(t_plot), 3, 3); % examples(t, j) = [q_j,w_j,u_j]
        for k=1:length(t_plot)
            % x у нас уже найдено, находим управление u. 
            if isequal(u_func, @sdre)
                % SDRE 
                u = u_func(X(k, 1:3)', X(k, 4:end)', w_0, I, I_inv, Q, R);
            else
                % NFC 
                u = u_func(X(k, :));
            end
            % сохраняем (x,u)
            examples(k, :, :) = [X(k, 1:3)', X(k, 4:6)', u];
        end
        res = examples;
    end
end

function dXdt = spacecraft_right_hand_sides(t, x, u_func, w_0, I, I_inv, Q, R)
    w_br = x(1:3);
    q = x(4:end);
    
    C = [1 - 2*(q(2)^2 + q(3)^2), 2*(q(1)*q(2) + q(3)*q(4)), 2*(q(1)*q(3) - q(2)*q(4));
         2*(q(1)*q(2) - q(3)*q(4)), 1 - 2*(q(1)^2 + q(3)^2), 2*(q(2)*q(3) + q(1)*q(4));
         2*(q(1)*q(3) + q(2)*q(4)), 2*(q(2)*q(3) - q(1)*q(4)), 1 - 2*(q(1)^2 + q(2)^2)];
    w_ri = [0; -w_0; 0];
    
    if isequal(u_func, @sdre)
        u = u_func(w_br, q, w_0, I, I_inv, Q, R);
    else
        u = u_func(x);
    end
    
    w_br_dot = -cross(C*w_ri, w_br) - I_inv*cross(w_br, I*w_br) ...
        + I_inv*cross(I*C*w_ri, w_br) - I_inv*cross(C*w_ri, I*w_br) ...
        - I_inv*cross(C*w_ri, I*C*w_ri) + I_inv*u;   
    
    q_dot = 0.5 * ( q(4) * w_br - cross(w_br, q(1:3)) );
    q_dot(4) = -0.5 * dot(w_br, q(1:3));
    
    dXdt = [w_br_dot; q_dot];
end

function u = sdre(w_br, q, w_0, I, I_inv, Q, R)
    C2 = [2*(q(1)*q(2) + q(3)*q(4)); 
          1 - 2*(q(1)^2 + q(3)^2); 
          2*(q(2)*q(3) - q(1)*q(4))];
    r = 1 / (2 * (q(1) + q(2) + q(3)));

    A11 = w_0 * cross_matrix(C2) - I_inv * cross_matrix(w_br) * I ...
        - w_0 * I_inv * cross_matrix(I * C2) + ...
        w_0 * I_inv * cross_matrix(C2) * I;
    A12 = -2 * w_0^2 * I_inv * cross_matrix(C2) * I ...
        * [q(2), 0, q(4);-q(1) + r, r, -q(3) + r; -q(4) q(3) 0];
    A21 = 0.5 * q(4) * eye(3);
    A22 = -0.5 * cross_matrix(w_br);
    
    A = [A11 A12; A21 A22];
    B = [I_inv; zeros(3)];
    [P, L, K] = care(A, B, Q, R);
    u = -K * [w_br; q(1:3)];
end

function m = cross_matrix(vec)
    m = [ 0      -vec(3)  vec(2);
          vec(3)  0      -vec(1);
         -vec(2)  vec(1)  0];
end