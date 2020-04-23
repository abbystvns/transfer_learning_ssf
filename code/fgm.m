function [sigma,error,error1,error2] = fgm(n,p)
X = normrnd(0,1,[n,p]);
S = normrnd(0,1,[n,n]);
lam = svd(X.'*X);
L = (max(lam))^2;
mu = (min(lam))^2;
q = mu/L;
alpha1 = 0.5;
[sigma,r] = psdp(X,S);
Y = sigma;
T = 40;
for i  = 1:T
    sigma0 = sigma;
    G = X.'*X*Y*X.'*X - X.'*S*X;
    B = Y - G/L;
    [Z,D] = eig((B + B.')/2);
    D1 = abs(D);
    D = (D+D1)/2;
    sigma = Z*D*(Z.');
    alpha = (q - alpha1^2 + sqrt((q - alpha1)^2 + 4 * alpha1^2))/2;
    beta = alpha1*(1-alpha1)/(alpha1^2 + alpha);
    alpha1 = alpha;
    Y = sigma + beta*(sigma - sigma0);
    %error = norm(X*sigma*(X.') - S,'fro')
    if norm(sigma - sigma0, 'fro') < 10^(-4)
        break;
    end
end
error = norm(X*sigma*(X.') - S,'fro');
J = X*sigma*(X.') - S;
error1 = norm(J(1:r,1:r),'fro');
J(1:r,1:r) = zeros(r);
error2 = norm(J,'fro');

function [sigma,rs] = psdp(X,S)
ns = length(S);
[~,ps] = size(X);
[U,Lambda,V] = svd(X);
rs = nnz(Lambda);
K = Lambda(1:rs,1:rs);
S = U.'*S*U;
S11 = S(1:rs,1:rs);
B_ = (S11 + S11.')/2;
[Z_,D_] = eig(B_);
D1_ = abs(D_);
D_ = (D_+D1_)/2;
sigma11 = Z_*D_*(Z_.');
sigma11 = inv(K)*sigma11*inv(K);
sigma = V.'*blkdiag(sigma11,eye(ps-rs))*V;%one example that sigma is pd,
                                         %cant determine the rest of 3 block matrix
end
end
% I've tried n = 50, p  = 5/10/30, and it all gives the results that 
% when n >> p
% most of the error comes from the rest 3 block matrices(dig out the first r*r matrix)
% Meanwhile, even error from the first r*r matrix can not be narrowed down to a very
% small value since from Higham's paper the error is always larger than the
% sum of square of eigenvalues which are less than zero.(page 106)
% Here the error refers to norm(X*sigma*(X.') - S,'fro').