clc;  clear;   close all;
% SBL algorithm

% signal generation
n = 512;                % signal dimension
rate = 4;
m = n*rate;
prior_pi = 0.1;
prior_mean = 0;
prior_var = 1/0.1;
Afro2 = n;
SNR = 50;
lar_num = 1e8;
sma_num = 1e-8;
tol = 1e-3;
T = 100;    % maximum number of iterations
NMSE_SBL = zeros(1,T);

lar_pos = 1e8;
sma_pos = 1e-8;    

z_A_ext = zeros(m,1);
v_A_ext = lar_pos;

a = 0.5;
b = 1e-3;
maxit_inner = 1;

x = zeros(n,1);
tau = zeros(m,1);
supp = find(rand(n,1)<prior_pi);
K = length(supp);
x(supp) = prior_mean + sqrt(prior_var)*randn(K,1);
A = randn(m,n);
A = sqrt(Afro2/trace(A'*A))*A;
z = A*x;
wvar = (z'*z)*10^(-SNR/10)/m;
w = sqrt(wvar)*randn(m,1);
y = sign(z+w+tau);

mu = zeros(size(x));
Sigma = lar_pos*eye(n);
    for iter_sbl = 1:T
        v_A_ext = lar_pos.*(v_A_ext<0)+v_A_ext.*(v_A_ext>0);
        [z_B_post, v_B_post] = GaussianMomentsComputation(y, 0, z_A_ext, v_A_ext*ones(m,1), wvar);
        v_B_post = mean(v_B_post);
        v_B_ext = v_B_post.*v_A_ext./(v_A_ext-v_B_post);
        z_B_ext = v_B_ext.*(z_B_post./v_B_post-z_A_ext./v_A_ext);

        beta = 1/v_B_ext;
        wvar0 = v_B_ext;
        y_tilde = z_B_ext;

        for k = 1:maxit_inner
            alpha = (1+2*a)./(mu.*mu+diag(Sigma)+2*b); 
            Sigma = inv(beta*(A'*A)+diag(alpha));
            mu = beta*Sigma*A'*y_tilde;           
        end
        z_A_post = A*mu;
    %   v_A_post = sum(s_bar_ext.^2./(gamma_2k+gamma_w.*s_bar_ext.*s_bar_ext))/M;
        v_A_post = mean(diag(A*Sigma*A'));
        v_A_ext = v_A_post.*wvar0./(wvar0-v_A_post);
        z_A_ext = v_A_ext.*(z_A_post./v_A_post-y_tilde./wvar0);
        c = mu'*x/(mu'*mu);
        NMSE_SBL(iter_sbl) = 20*log10(norm(c*mu-x)/norm(x));
    end

iteration = 1:T;
figure(1)
plot(iteration,NMSE_SBL,'-bv');
xlabel('iteration')
ylabel('debiased NMSE')
legend('GrSBL')
