function [xhat, vx, dMSE] = Gramp( A, y, tau, pi, mu, sigvar, wvar, T2, x)
% GrAMP algorithm for one bit compressed sensing under additive Gaussian
% noise

% Input:
% - A: measurement matrix (m x n)
% - y: sign measurements (+1 or -1) (m x 1)
% - tau: quantizer thresholds
% - init: initialization for the signal and variance [xhat0, vx0]
% - pi0, mu0, tau0, Delta0: initialized prior nonzero probability, prior mean, prior
% variance, additive noise variance
% - T: number of iterations
% - wvar: variance of additive noise
% Output:
% - xhat: reconstructed signal (n x 1)
% - vx: predicted MSE (n x 1)
% - dMSE: debiased MSE

% Number of measurements and dimension of the signal
global lar_num sma_num dampFac tol T counter
ct = counter;
[m, n] = size(A);
% absdiff = @(x) sum(abs(x))/length(x);
xhat = zeros(n,1);
vx = lar_num*ones(n, 1);

% Initialize shat
shat = zeros(m, 1);

% Previous estimate
xhatprev = xhat;
shatprev = shat;
vxprev = vx;

z_A_ext = zeros(m,1);
v_A_ext = lar_num*ones(m,1);

% Hadamard product of the matrix
AA = A.*A;
vp = lar_num*ones(m,1);
phat = zeros(m,1);
% Perform estimation
computeMse = @(noise) 20*log10(norm(noise(:))/norm(x));
absdiff = @(zvar) sum(abs(zvar))/length(zvar);

dMSE = zeros();
for t = 1:T
    if t == 1
        dMSE(t) = computeMse(x);
    else
        c0 = xhat'*x/(xhat'*xhat);
        dMSE(t) = computeMse(c0*xhat-x);
    end
    
    v_A_ext = lar_num*(v_A_ext<0)+v_A_ext.*(v_A_ext>0);
    v_A_ext = min(v_A_ext,lar_num);
    v_A_ext = max(v_A_ext,sma_num);
    [z_B_post, v_B_post] = GaussianMomentsComputation(y, tau, z_A_ext, v_A_ext, wvar);
    v_B_ext = v_B_post.*v_A_ext./(v_A_ext-v_B_post);
    v_B_ext = lar_num*(v_B_ext<0)+v_B_ext.*(v_B_ext>0);
    v_B_ext = min(v_B_ext,lar_num);
    v_B_ext = max(v_B_ext,sma_num);
    z_B_ext = v_B_ext.*(z_B_post./v_B_post-z_A_ext./v_A_ext+eps);
    xhatprev0 = xhat;
    for t2 = 1:T2
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Linear
        vs = 1./(v_B_ext + vp);
        shat = (z_B_ext-phat).*vs;
        vr = 1./(AA' * vs);
        rhat = xhat + vr .* (A' * shat); 

        % Non-linear variable estimation 
        M = 0.5*log(vr./(vr+sigvar))+0.5*rhat.^2./vr-0.5*(rhat-mu).^2./(vr+sigvar);
        lambda = pi./(pi+(1-pi).*exp(-M));
        m_t = (rhat.*sigvar+vr.*mu)./(vr+sigvar);
        V_t = vr.*sigvar./(vr+sigvar);
        
    % Compute E{X|Rhat = rhat}
        xhat = lambda.*m_t;

    % Compute Var{X|Rhat = rhat}
        vx = lambda.*(m_t.^2+V_t)-xhat.^2;

        %Damp
        xhat = dampFac*xhat + (1-dampFac)*xhatprev;
        shat = dampFac*shat + (1-dampFac)*shatprev;
        vx = dampFac*vx + (1-dampFac)*vxprev;
        
        % If without a change
        % Save previous xhat
        xhatprev = xhat;
        shatprev = shat;
        vxprev = vx;
        vp = AA*vx;
        phat = A*xhat - vp.*shat;
     end

     if(absdiff(xhat - xhatprev0) < tol)
        ct = ct - 1;
     end
     if(ct <= 0)
            break;
     end
     
     z_A_ext = phat;
     v_A_ext = vp;

        % Stopping criterion
    %     if(t>2&&abs(dMSE(t)-dMSE(t-1))<tol)
    %         break;
    %     end
        
end

end

