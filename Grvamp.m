function [x_hat_1k, x_hat_var_1k, dMSE] = Grvamp( A, y, tau, pi, xmean1, xvar1, wvar, T2, x, S_A, V_A, SA2, SAUA)
% GrVAMP algorithm for one bit compressed sensing under additive Gaussian
global lar_num sma_num T tol counter
[m, n] = size(A);
ct = counter;
% [U_A,S_A,V_A]=svd(A);
s_bar = diag(S_A);
s_bar_ext = [s_bar;zeros(n-length(s_bar),1)];
% GVAMP algorithm for one bit compressed sensing under additive Gaussian
z_A_ext = zeros(m,1);
v_A_ext = lar_num;
x_hat_1k= zeros(n,1); 
% x_hat_var_1k= lar_num*ones(n, 1);

gamma1k = sma_num;
gamma1k_inv = 1/gamma1k;
r_1k = zeros(size(x));

% Perform estimation
computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));
absdiff = @(zvar) sum(abs(zvar))/length(zvar);

dMSE = zeros();
for t = 1:T
    if t == 1
        dMSE(t) = computeMse(x);
    else
        c0 = x_hat_1k'*x/(x_hat_1k'*x_hat_1k+eps);
        dMSE(t) = computeMse(c0*x_hat_1k-x);
    end

    v_A_ext = lar_num*(v_A_ext<0)+v_A_ext*(v_A_ext>0);
    v_A_ext = min(v_A_ext,lar_num);
    v_A_ext = max(v_A_ext,sma_num);
    [z_B_post, v_B_post] = GaussianMomentsComputation(y, tau, z_A_ext, v_A_ext*ones(m,1), wvar);
    v_B_post = mean(v_B_post);        
    sigma2_tilde = v_B_post.*v_A_ext./(v_A_ext-v_B_post); % equantion (6)
    sigma2_tilde = lar_num*(sigma2_tilde<0)+sigma2_tilde.*(sigma2_tilde>0);
    sigma2_tilde = min(sigma2_tilde,lar_num);
    sigma2_tilde = max(sigma2_tilde,sma_num);
    y_tilde = sigma2_tilde.*(z_B_post./v_B_post-z_A_ext./v_A_ext+eps);  % equantion (7)

    gamma_w = 1./sigma2_tilde;
    x_hat_1k_pre = x_hat_1k;
    for t2 = 1:T2
        % denoising step
        L = -0.5*log(1+xvar1./gamma1k_inv)+r_1k.^2/2/gamma1k_inv-(r_1k-xmean1).^2/2/(gamma1k_inv+xvar1);
        m_1k= (xvar1*r_1k+gamma1k_inv*xmean1)/(gamma1k_inv+xvar1);
        post_prior=pi./(pi+(1-pi)*exp(-L));
        post_var = xvar1.*gamma1k_inv./(gamma1k_inv+xvar1);

        x_hat_1k = post_prior.*m_1k;
        x_hat_var_1k = post_prior.*(m_1k.^2+post_var)-(post_prior.*m_1k).^2;
        % averaging operation
        alpha_1k = mean(x_hat_var_1k*gamma1k);
        eta_1k = gamma1k/alpha_1k;
        gamma_2k = eta_1k-gamma1k;
        r_2k = (eta_1k/gamma_2k)*x_hat_1k-gamma1k/gamma_2k*r_1k;
        % LMMSE step
        diag_eq = gamma_w*SA2+gamma_2k+eps;
        x_hat_2k = V_A*diag(1./diag_eq)*(gamma_w*SAUA*y_tilde+gamma_2k*V_A'*r_2k);
%         x_hat_2k = V_A/(gamma_w*SA2+(gamma_2k+eps)*eye(n))*(gamma_w*SAUA*y_tilde+gamma_2k*V_A'*r_2k);
        alpha_2k = gamma_2k/n*sum(1./(gamma_2k+gamma_w.*s_bar_ext.*s_bar_ext));
        eta_2k = gamma_2k/alpha_2k;
    %     mean(eta_2k_diag)
        gamma1k = eta_2k-gamma_2k;
        gamma1k_inv = 1/gamma1k;
        r_1k = (eta_2k*x_hat_2k-gamma_2k*r_2k)/gamma1k;
    end
    z_A_post = A*x_hat_2k;
%     v_A_post = 1/m*trace(A/(gamma_w*(A'*A)+gamma_2k*eye(n))*A');%   
    v_A_post = sum(s_bar_ext.^2./(gamma_2k+gamma_w.*s_bar_ext.*s_bar_ext))/m;
    v_A_ext = v_A_post.*sigma2_tilde./(sigma2_tilde-v_A_post);
    v_A_ext = mean(v_A_ext);
    z_A_ext = v_A_ext.*(z_A_post./v_A_post-y_tilde./sigma2_tilde);
%     c = x_hat_1k'*x/(x_hat_1k'*x_hat_1k);
            % Stopping criterion
      if(absdiff(x_hat_1k_pre-x_hat_1k)<tol)
            ct = ct-1;
      end
      if(ct<=0)
            break;
      end
%     if(t>2&&abs(dMSE(t)-dMSE(t-1))<tol)
%         break;
%     end
end
end
