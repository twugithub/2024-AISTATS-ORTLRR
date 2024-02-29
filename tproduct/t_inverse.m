function [ C ] = t_inverse( A, transform )

% The inverse of a 3 way tensor (A'*A+I) under linear transform
%
% Input:
%       A       -   n1*n2*n3 tensor
%   transform   -   a structure which defines the linear transform
%       transform.L: the linear transform of two types:
%                  - type I: function handle, i.e., @fft, @dct
%                  - type II: invertible matrix of size n3*n3
%
%       transform.inverseL: the inverse linear transform of transform.L
%                         - type I: function handle, i.e., @ifft, @idct
%                         - type II: inverse matrix of transform.L
%
%       transform.l: a constant which indicates whether the following property holds for the linear transform or not:
%                    L'*L=L*L'=l*I, for some l>0.
%                  - transform.l > 0: indicates that the above property holds. Then we set transform.l = l.
%                  - transform.l < 0: indicates that the above property does not hold. Then we can set transform.l = c, for any constant c < 0.
%       If not specified, fft is the default transform, i.e.,
%       transform.L = @fft, transform.l = n3, transform.inverseL = @ifft.
%
%
% Output:
%        C      -   inverse of tensor (A'*A+I)
%
%
%
% See also lineartransform, inverselineartransform

[~, n2, n3] = size(A);
if nargin < 2
    % fft is the default transform
    transform.L = @fft; transform.l = n3; transform.inverseL = @ifft;
end

if isequal(transform.L,@fft)
    % efficient computing for fft transform
    A = fft(A,[],3);
    C = zeros(n2,n2,n3);
    for i = 1:n3
        C(:,:,i) =  (A(:,:,i)'*A(:,:,i) + eye(n2))\eye(n2);
    end
    C = ifft(C,[],3);
else
    % other transform
    A = lineartransform(A,transform);
    C = zeros(n2,n2,n3);
    for i = 1:n3
        C(:,:,i) =  (A(:,:,i)'*A(:,:,i) + eye(n2))\eye(n2);
    end
    C = inverselineartransform(C,transform);
end

end