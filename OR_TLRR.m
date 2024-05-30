function [ Z, E, Z_tnn, err ] = OR_TLRR( X, A, lambda, transform, max_iter )

[n1, n2, n3] = size(X);
[~, n4, ~] = size(A);

%% initialize variables
Z = zeros(n4,n2,n3);
J = Z;
Y1 = Z;
E = zeros(n1,n2,n3);
Y2 = E;

beta = 1e-4;
max_beta = 1e+8;
tol = 1e-8;
rho = 1.1;
iter = 0;

Ain = t_inverse(A, transform);
AT = tran(A, transform);

while iter < max_iter
    iter = iter+1;

    %% update Zk
    Z_pre = Z;
    R1 = J - Y1/beta;
    [Z,Z_tnn] = prox_tnn(R1, 1/beta, transform);

    %% update Ek
    E_pre = E;
    R2 = X - tprod(A, J, transform) + Y2/beta;
    E = prox_l21(R2, lambda/beta);

    %% update Jk
    J_pre = J;
    Q1 = Z + Y1/beta;
    Q2 = X - E + Y2/beta;
    J = tprod(Ain, Q1 + tprod(AT, Q2, transform), transform);

    %% check convergence
    leq1 = Z - J;
    leq2 = X - tprod(A, J, transform) - E;
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));

    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    err = max([leqm1,leqm2,difJ,difZ,difE]);
    if (iter==1 || mod(iter,20)==0)
        fprintf('iter = %d, err = %.8f\n',iter,err);
    end
    if err < tol
        break;
    end

    %% update Lagrange multiplier and penalty parameter beta
    Y1 = Y1 + beta*leq1;
    Y2 = Y2 + beta*leq2;
    beta = min(beta*rho,max_beta);
end

end