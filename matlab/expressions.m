function expressions()
    I_inv = inv(eye(3));
    I = eye(3);
    
    w_0 = sym('w_0');
    w = sym('w', [3 1]);
    q = sym('q', [4 1]);
    A11 = sym('A11', [3 3]);
    A12 = sym('A12', [3 3]);
    A21 = sym('A21', [3 3]);
    A22 = sym('A22', [3 3]);
    u = sym('u', [3 1]);
    C = sym('C', [3 3]);

    C2 = [2*(q(1)*q(2) + q(3)*q(4)); 
          1 - 2*(q(1)^2 + q(3)^2); 
          2*(q(2)*q(3) - q(1)*q(4))];
    r = 1 / (2 * (q(1) + q(2) + q(3)));
    A11 = w_0 * cross_matrix(C2) - I_inv * cross_matrix(w) * I ...
            - w_0 * I_inv * cross_matrix(I * C2) + ...
            w_0 * I_inv * cross_matrix(C2) * I;
    A12 = -2 * w_0^2 * I_inv * cross_matrix(C2) * I ...
            * [q(2), 0, q(4);-q(1) + r, r, -q(3) + r; -q(4) q(3) 0];
    A21 = 0.5 * q(4) * eye(3);
    A22 = -0.5 * cross_matrix(w);
    A = [A11 A12; A21 A22];
    B = [I_inv; zeros(3)];
    
    x_dot = A * [w; q(1:3)] + B * u;

end

function m = cross_matrix(vec)
    m = [ 0      -vec(3)  vec(2);
          vec(3)  0      -vec(1);
         -vec(2)  vec(1)  0];
end