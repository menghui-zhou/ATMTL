function [W, funcVal] = update_W_classify(W_warm, X, Y, A, B, C, D,  R, N, opts)

opts = init_opts(opts);
rho = opts.rho;
task_num  = length(X);
dimension = size(X{1}, 2);
funcVal = [];

W0 = W_warm;  % warm start
% pFlag = 0;  % parallel
% stopping criteria

Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

bFlag = 0;

while iter < opts.maxIter
    
    alpha = (t_old - 1) /t;
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
        
   while true
%       Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma, rho2 / gamma, rho3 / gamma);
        Wzp = Ws - gWs/gamma;  % No projection
        Fzp = funVal_eval(Wzp);
        
        delta_Wzp = Wzp - Ws;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        r_sum = nrm_delta_Wzp;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs))...
            + gamma/2 * nrm_delta_Wzp;
        
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
    
    
   Wz_old = Wz;
   Wz = Wzp; 
   
%    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1, rho2, rho3));
   funcVal = cat(1, funcVal, Fzp); % no nonsmooth
   
   
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
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
W = Wzp;



% private functions
    function [funcVal] = funVal_eval (W)
        log_fVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                log_fVal = log_fVal + unit_funcVal_eval( W(:, i), X{i}, Y{i});
            end
        else
            for i = 1: task_num
                log_fVal = log_fVal + unit_funcVal_eval( W(:, i), X{i}, Y{i});
            end
        end
        firstp = trace(  C' * (W*R - A)  )  + trace( D' * (W*N - B)  );
        secondp = rho/2 * (  norm(W*R-A, 'fro')^2 + norm(W*N-B, 'fro')^2  );
        funcVal = log_fVal + firstp + secondp;
        
        
    end

    function [grad_W] = gradVal_eval(W)   % logistic loss gradient
        log_grad = zeros(dimension, task_num);
        if opts.pFlag
            parfor i = 1:task_num
                [ log_grad(:, i)] = unit_grad_eval( W(:, i), X{i}, Y{i} );
            end
        else
            for i = 1:task_num
                [ log_grad(:, i)] = unit_grad_eval( W(:, i), X{i}, Y{i} );
            end
        end
        mat_grad = C*(R') +  rho * (W*R - A) * (R'); 
        mat_grad = mat_grad + D *(N') + rho * (W*N - B) * (N');
        grad_W = log_grad + mat_grad;
    end


end


function [ grad_w, funcVal ] = unit_grad_eval( w, x, y )
%Logistic_Eval. evaluation of logistic loss.
m = length(y);
% m = 1;
weight = ones(m, 1)/m;
% weight = ones(m, 1);
weighty = weight.* y;
aa = -y.*(x *w );
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_w = x' * b;
end



function [ funcVal ] = unit_funcVal_eval( w, x, y)
%function value evaluation for each task
m = length(y);

weight = ones(m, 1)/m;
% weight = ones(m, 1);
aa = -y.*(x*w);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end
