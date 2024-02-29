function x = prox_l1_miss(b, Xmiss, lambda)

observeidx = ~isnan(Xmiss);
x = b;
x(observeidx) = max(0,b(observeidx)-lambda)+min(0,b(observeidx)+lambda);
end