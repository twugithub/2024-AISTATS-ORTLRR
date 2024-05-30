clearvars;
close all;
clc

rng('shuffle');
addpath('tproduct');

n1 = 60;
n3 = 100;
clustern = 5;
rank_ratio = 0.1;
eta = 0.9;

transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
% transform.L = @dct; transform.l = 1; transform.inverseL = @idct;
% transform.L = RandOrthMat(n3); transform.l = 1;

all_n = n1 * ones(clustern, 1);
all_d = round(n1 * rank_ratio * ones(clustern, 1));
rho = 0.2;

A_gt = cell(1,clustern);
B_gt = cell(1,clustern);

for kidx = 1:clustern
    d = all_d(kidx);
    n = all_n(kidx);
    A_gt{kidx} = sqrt(1/n1) * randn(n1,d,n3);
    B_gt{kidx} = sqrt(1/n1) * randn(n,d,n3);
end

end_idx = cumsum(all_n);
start_idx = end_idx - all_n + 1;
n2 = sum(all_n);
L0 = zeros(n1,n2,n3);
for kidx = 1:clustern
    L0(:, start_idx(kidx):end_idx(kidx), :) = tprod(A_gt{kidx}, tran(B_gt{kidx}, transform), transform);
end
lnorm_avg = mean(sum(sum(L0.^2,3),1));

outlieridx = find(rand(n2,1) < rho);
numOutliers = length(outlieridx);
inlieridx = setdiff(1:n2, outlieridx);
E0 = zeros(n1,n2,n3);
E0(:,outlieridx,:) = sqrt(lnorm_avg/(n1*n3)) * randn(n1, numOutliers, n3);
L0(:,outlieridx,:) = 0;
X_noise = L0 + E0;

X_noise_miss = NaN(size(X_noise));

for i = 1:size(X_noise,2)
    tempdata = squeeze(X_noise(:,i,:));
    chosen = randperm(size(tempdata,1)*size(tempdata,2), round(eta*size(tempdata,1)*size(tempdata,2)));
    temp2 = NaN(size(tempdata,1),size(tempdata,2));
    temp2(chosen) = tempdata(chosen);
    X_noise_miss(:,i,:) = reshape(temp2, [size(tempdata,1),1,size(tempdata,2)]);
end

missidx = isnan(X_noise_miss);
X_noise_zf = X_noise;
X_noise_zf(missidx) = 0;

tho = 100;
[ ~,~,U_x,V_x,S_x ] = prox_low_rank(X_noise_zf,tho,transform);
LL = tprod(U_x,S_x,transform);

lambda = 2 / (sqrt(log(max(n1,n2)))*tsn(X_noise_zf,transform));
% FFT: lambda = 2 / (sqrt(log(max(n1,n2)))*tsn(X_noise_zf,transform))
% DCT/ROM: lambda = 30 / (sqrt(log(max(n1,n2)))*tsn(X_noise_zf,transform))

max_iter = 800;
[ Z, tlrr_E, X_rec, Z_rank, err_va ] = OR_TLRR_EWZF(X_noise_miss, LL, lambda, transform, max_iter);

E_norm = sum(sum(tlrr_E.^2,3),1);
[idx,c] = kmeans(E_norm', 2, 'Start', 'Plus');
if c(1) < c(2)
    pred_outlier_idx = find(idx==2);
    pred_normal_idx = find(idx==1);
else
    pred_outlier_idx = find(idx==1);
    pred_normal_idx = find(idx==2);
end

gt_onehot = zeros(n2,1);
pred_onehot = zeros(n2,1);
gt_onehot(outlieridx) = 1;
pred_onehot(pred_outlier_idx) = 1;
hamdist = sum(xor(gt_onehot, pred_onehot));