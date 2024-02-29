function [ Q, sparsity, supp_set, l21 ] = prox_l21_miss( Q, Xmiss, rho )

[n1, n2, n3] = size(Q);
sparsity = 0;
supp_set = [];
l21 = 0;
for i = 1:n2
    qsq = squeeze(Q(:,i,:));
    qsq = qsq(:);
    xsq = squeeze(Xmiss(:,i,:));
    missidx = isnan(xsq(:));
    tempq = qsq;
    tempq(missidx) = 0;
    QF = sum(tempq.*tempq)^0.5;
    if QF > rho
        ratio = 1-rho/QF;
        msq = ratio*tempq;
        msq(missidx) = qsq(missidx);
        sparsity = sparsity + 1;
        supp_set = [supp_set,i];
        l21 = l21 + ratio*QF;
    else
        msq = zeros(length(qsq),1);
        msq(missidx) = qsq(missidx);
    end
    Q(:,i,:) = reshape(msq, [n1, 1, n3]);
end
end