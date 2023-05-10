function dataset = get_dataset(n0, w_range, q_range)
    if nargin == 0
        n0 = 200;
        w_range = [-1, 1];
        q_range = [-0.2618, 0.2618];
    end
    % q_init = rand(4, n0) * (q_range(2) - q_range(1)) + q_range(1);
    w_init = rand(3, n0) * (w_range(2) - w_range(1)) + w_range(1);
    q_init = zeros(4, n0);
    k = 0;
    while k < n0
        % Нужно, чтобы l1^2 + l2^2 + l3^2 = 1.      
        l = zeros(3, 1); 
        l(1) = 2 * rand() - 1; % от -1 до 1  - это допустимые значения. 
        l(2) = sqrt(1 - l(1) ^ 2);
        l(2) = 2 * l(2) * rand() - l(2);
        l(3) = sqrt(1 - l(1) ^ 2 - l(2) ^ 2);
        if rand() > 0.5
            l(3) = -l(3);
        end
        l(:) = l(randperm(3)); % случайная перестановка l1, l2, l3. 
        phi = 2 * pi * rand(); % случайный угол от 0 до 2pi. 
        new_q = [l * cos(phi/2); sin(phi/2)]; 
        
        if (sum(new_q.^2, 'all') == 1) & (all(q_range(1) <= new_q(1:3))) & (all(new_q(1:3) <= q_range(2)))
            k = k + 1;
            q_init(:, k) = new_q;
        end
    end
    
    agg_data = u_data(0, w_init(:, 1), q_init(:, 1));
    sz = size(agg_data);
    dataset = zeros(n0 * sz(1), sz(2), sz(3));
    dataset(1 : sz(1), :, :) = agg_data;
    for k=2:n0
        agg_data = u_data(0, w_init(:, k), q_init(:, k));
        dataset((k-1)*sz(1)+1 : k*sz(1), :, :) = agg_data;
    end
end