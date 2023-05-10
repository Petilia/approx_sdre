function q_init = test()
    if nargin == 0
        n0 = 50;
        w_range = [-1, 1];
        q_range = [-0.2618, 0.2618];
    end
    % q_init = rand(4, n0) * (q_range(2) - q_range(1)) + q_range(1);
    w_init = rand(3, n0) * (w_range(2) - w_range(1)) + w_range(1);
    q_init = zeros(4, n0);
    k = 0;
    while k < n0
        l = zeros(3, 1);
        l(1) = 2 * rand() - 1;
        l(2) = sqrt(1 - l(1) ^ 2);
        l(2) = 2 * l(2) * rand() - l(2);
        l(3) = sqrt(1 - l(1) ^ 2 - l(2) ^ 2);
        if rand() > 0.5
            l(3) = -l(3);
        end
        l(:) = l(randperm(3));
        phi = 2 * pi * rand();
        new_q = [l * cos(phi/2); sin(phi/2)]; 
        
        if (sum(new_q.^2, 'all') == 1) & (all(q_range(1) <= new_q(1:3))) & (all(new_q(1:3) <= q_range(2)))
            k = k + 1;
            q_init(:, k) = new_q;
        end
    end
end