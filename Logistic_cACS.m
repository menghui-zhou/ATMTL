function [W_end, C, funcVal] = Logistic_cACS(X, Y, rho1, rho2, rho3, alpha, opts)

if nargin < 7
    opts = [];
end

% sig_max = max_singular_MTL(X) ^ 2;
opts = init_opts(opts);

task_num  = length (X);

% W*R = Q, W = Q*inv(R) = Q*S;
R = adaptive_correlation(alpha, task_num);  % adaptive correlation among tasks
S = pinv(R);

% temporal relation matrix
H = zeros(task_num, task_num - 1);
H(1 : (task_num + 1) : end) = 1;
H(2 : (task_num + 1) : end) = -1;
F = H';  % transition matrix


dimension = size(X{1}, 2);  % feature number
funcVal = []; % function value


Q0 = zeros(dimension, task_num);
C0 = zeros(1, task_num);



bFlag=0; % this flag tests whether the gradient step only changes a little


Qz = Q0;
Cz = C0;
Qz_old = Q0;
Cz_old = C0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
% gamma = sig_max;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Qs = (1 + alpha) * Qz - alpha * Qz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    
    % compute function value and gradients of the search point
    [gQs, gCs, Fs ]  = gradVal_eval(Qs, Cs);
    
    while true
        Qzp = FGLasso_projection(Qs - gQs/gamma, rho1 / gamma, rho2 / gamma, rho3 / gamma);
        Czp = Cs - gCs/gamma;
        Fzp = funVal_eval  (Qzp, Czp);
        
        delta_Qzp = Qzp - Qs;
        delta_Czp = Czp - Cs;
        nrm_delta_Qzp = norm(delta_Qzp, 'fro')^2;
        nrm_delta_Czp = norm(delta_Czp, 'fro')^2;
        r_sum = (nrm_delta_Qzp + nrm_delta_Czp)/2;
        
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + trace(delta_Czp' * gCs)...
        %             + gamma/2 * norm(delta_Wzp, 'fro')^2 ...
        %             + gamma/2 * norm(delta_Czp, 'fro')^2;
        
        Fzp_gamma = Fs + sum(sum(delta_Qzp .* gQs))...
            + sum(sum(delta_Czp .* gCs))...
            + gamma/2 * nrm_delta_Qzp ...
            + gamma/2 * nrm_delta_Czp;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Qz_old = Qz;
    Cz_old = Cz;
    Qz = Qzp;
    Cz = Czp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Qz, rho1, rho2, rho3));
    
%     if (bFlag)
%         % fprintf('\n The program terminates as the gradient step changes the solution very small.');
%         break;
%     end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

Q = Qzp;
C = Czp;
W_end = Q * S;
% private functions

    function [Qp] = FGLasso_projection (Q, lambda_1, lambda_2, lambda_3 )
       
        Qp = zeros(size(Q));
        
        for i = 1 : size(Q, 1)
            v = Q(i, :);
            q = FGLasso_projection_rowise(v, lambda_1, lambda_2, lambda_3);
            Qp(i, :) = q';
        end
    end

% smooth part gradient and function value. .
    function [grad_Q, grad_C, funcVal] = gradVal_eval(Q, C)
        grad_Q = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);

        for i = 1:task_num
            [ grad_Q_temp, grad_C(:, i), lossValVect(:, i)] = ...
                                  unit_grad_eval(Q, S(:, i), C(i), X{i}, Y{i} );
            grad_Q = grad_Q + grad_Q_temp;
%             grad_C = grad_C + grad_C_temp;
        end

%         %grad_W = grad_W+ rho1 * 2 *  W * RRt;
        funcVal = sum(lossValVect);% + rho1 * norm(W*R, 'fro')^2;

    end

% smooth part function value.
    function [funcVal] = funVal_eval (Q, C)
        funcVal = 0;
        W = Q * S;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            for i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end

    end

% non-smooth part function value.
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



function [ grad_Q, grad_c, funcVal ] = unit_grad_eval(Q, s, c, x, y )
%Logistic_Eval. evaluation of logistic loss.
w = Q * s;
m = length(y); % sample
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x * w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_c = sum(b);
% grad_Q =  x' * s' * b;

grad_Q = zeros(size(Q));
for i = 1 : m
    grad_Q = grad_Q + (x(i,:)' * (s')) * b(i);
end
end

% function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y )
% %Logistic_Eval. evaluation of logistic loss.
% m = length(y);
% weight = ones(m, 1)/m;
% weighty = weight.* y;
% aa = -y.*(x'*w + c);
% bb = max( aa, 0);
% funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
% pp = 1./ (1+exp(aa));
% b = -weighty.*(1-pp);
% grad_c = sum(b);
% grad_w = x * b;
% end



function [ funcVal ] = unit_funcVal_eval( w, c, x, y)
%function value evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
aa = -y.*(x*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end