function [W_end, funcVal] = Least_cACS(X, Y, rho1, rho2, rho3, alpha, opts)

if nargin < 7
    opts = [];
end

sig_max = max_singular_MTL(X) ^ 2;

opts = init_opts(opts);  

task_num  = length (X);    

% W*R = Q, W = Q*inv(R) = Q*S;
R = adaptive_correlation(alpha, task_num);  % adaptive correlation among tasks
S = pinv(R);

% S = pinv(adaptive_correlation(alpha, task_num));


% temporal relation matrix
H = zeros(task_num, task_num - 1);
H(1 : (task_num + 1) : end) = 1;
H(2 : (task_num + 1) : end) = -1;
F = H';  % transition matrix


dimension = size(X{1}, 2);  % feature number
funcVal = []; % function value


Q0 = zeros(dimension, task_num);

bFlag = 0; % this flag tests whether the gradient step only changes a little


Qz = Q0;
Qz_old = Q0;

t = 1;
t_old = 0;

iter = 0;
gamma = sig_max;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Qs = (1 + alpha) * Qz - alpha * Qz_old;  % search point

    % compute function value and gradients of the search point
    gQs  = gradVal_eval(Qs, S);  
    Fs   = funVal_eval(Qs);
    
    while true
        % approximate point
        Qzp = FGLasso_projection(Qs - gQs/gamma, rho1 / gamma, rho2 / gamma, rho3 / gamma);
        Fzp = funVal_eval(Qzp); 
        
        
               
        % augmented function value
        % Fzp_gamma = Fzp_augmented at x point = 
        % F(x) + <s - x, gradient(x)) + rho / 2 * ||x-s||_2^2
        delta_Qzp = Qzp - Qs;
        nrm_delta_Qzp = norm(delta_Qzp, 'fro')^2;
        residual_sum = nrm_delta_Qzp;
        Fzp_gamma = Fs + sum(sum(delta_Qzp .* gQs))...
            + gamma/2 * nrm_delta_Qzp;
        
        if (residual_sum <= 1e-20)
            bFlag = 1; % this shows that, the difference between approximate
                     % point and search point is very small.
            break;
        end      
        
        % satisfy the break rule
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
        
        
    end
    
    
    Qz_old = Qz;
    Qz = Qzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Qz, rho1, rho2, rho3));
    
    if (bFlag)
         fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            
            if iter>=2
                if (abs(funcVal(end) - funcVal(end-1)) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs(funcVal(end) - funcVal(end-1)) <=...
                        opts.tol * funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end) <= opts.tol)
                break;
            end
        case 3
            if iter >= opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

Q = Qzp;
% W = Q * S;
W_end = Q * S;

% private functions

    function [Qp] = FGLasso_projection (Q, lambda_1, lambda_2, lambda_3)
        
        Qp = zeros(size(Q));
        
        for i = 1 : size(Q, 1)
            v = Q(i, :);
            q = FGLasso_projection_rowise(v, lambda_1, lambda_2, lambda_3);
            Qp(i, :) = q';
        end
    end



% smooth part gradient.
    function [grad_Q] = gradVal_eval(Q, S)

        grad_Q = zeros(size(Q));
        for  i = 1 : task_num
            grad_Q = grad_Q + X{i}' * (X{i} * Q * S(:,i) - Y{i}) * S(:,i)';
        end

    end


% smooth part function value.
    function [funcVal] = funVal_eval(Q)
        
        funcVal = 0;
         W = Q * S;
        if opts.pFlag
            parfor i = 1 : task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i} * W(:, i))^2;
            end
        else
            for i = 1 : task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i} * W(:, i))^2;
            end
        end
    end




    function [non_smooth_value] = nonsmooth_eval(Q, rho_1, rho_2, rho_3)
        non_smooth_value = 0;
        for i = 1 : size(Q, 1)
            q = Q(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(q, 1) + rho_2 * norm(F * q', 1) ...
                + rho_3 * norm(q, 2);
        end
    end



end