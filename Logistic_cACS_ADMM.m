function [W, last_fval, out] = Logistic_cACS_ADMM(X, Y, lambda1, lambda2, lambda3, alpha, opts)

if nargin < 7
    opts = [];

end

opts = init_opts(opts);
% opts.tFlag = 0;
% opts.maxIter = 1000;
% opts.tol = 1e-4;
% opts.fFlag = 1; % relative change

if ~isfield(opts, 'p_residual'); opts.p_residual = 1e-3; end  % primal residual tolerance
if ~isfield(opts, 'd_residual'); opts.d_residual = 1e-3; end  % dual residual tolerance
if ~isfield(opts, 'rho');        opts.rho = 0.1;     end  % the parameter of quadratic penalty
if ~isfield(opts, 'tao');        opts.tao = 1.618;    end  % adjust the step size, (0, (1+sqrt(5))/2)
if ~isfield(opts, 'verbose');    opts.verbose = 0;    end  % display middle result


R = adaptive_correlation(alpha, length(X));
task_num = length(X);
H = zeros( task_num, task_num-1 );
H(1 : (task_num+1) : end) = 1;
H(2 : (task_num+1) : end) = -1;



iter = 1;
tt = tic;
dim = size(X{1}, 2); 
W = zeros(dim, task_num);  
out = struct();



fp = inf;
primal_residual = inf; %  primal_residual
dual_residual = inf;   %  dual_residual 
f = func_Value(X, Y, W, lambda1, lambda2, lambda3, R, H);   
f0 = f;         
out.fvec = f0;  



A = zeros(size(W));
B = zeros(size(W, 1), size(H, 2));
% Dual variables
C = zeros(size(A));
D = zeros(size(B));



N = R * H;



while  iter < opts.maxIter 
    
    W = update_W_classify(W, X, Y, A, B, C, D, R, N, opts);


    A_old = A;
    A_target = W * R + C / opts.rho;
    A = prox_SGL(A_target, lambda1 / opts.rho, lambda2 / opts.rho);
    

    B_old = B;
    B_target = W * N + D / opts.rho;
    B = prox_FL(B_target, lambda3 / opts.rho);
    
    
    dr = opts.rho * ((A - A_old) + (B - B_old) * H');
    dual_residual = norm(dr, 'fro');
    
   
    C = C + opts.tao * opts.rho * (W*R - A);
    D = D + opts.tao * opts.rho * (W*R*H - B);
    
    fp = f;
    f = func_Value(X, Y, W, lambda1, lambda2, lambda3, R, H);
    
    pr1 = norm(W * R- A, 'fro');
    pr2 = norm(W*R*H - B, 'fro');
    primal_residual = pr1 + pr2;
    
    

    if opts.verbose
        fprintf('iter: %d \t fval: e%e \t feasi: %.1e \n', iter, f, primal_residual);
    end
    
    iter = iter + 1; 
    out.fvec = [out.fvec; f];
    

    switch(opts.tFlag)
        case 0
            if iter >= 2
                if (abs(f - fp) <= opts.tol); break; end
            end
        case 1
            if iter >= 2
                
                if (abs(f - fp) < fp * opts.tol)
                    break;
                end
            end
        case 2
            if iter >= 2
                if f <= opts.tol; break; end
            end
        case 3
            if iter >= opts.maxIter; break; end
    end
    
  
end



    out.W_candidate = A;
    out.C = C;
    out.D = D;
    
    out.end_fval = f;
    out.end_iter = iter;
    out.tt = toc(tt);
    out.primal_residual = primal_residual; 
    out.dual_residual = dual_residual;
    out.f_delta = abs(fp - f);
    last_fval = out.fvec;

end



function val = func_Value(X, Y, W, lambda1, lambda2, lambda3, R,  H)

    part1 = 0;
    for i = 1 : length(X)
        part1 = part1 + unit_funcVal_eval( W(:, i), X{i}, Y{i});
    end
   

    W = W * R;
    part2 = 0;
    for i = 1 : size(W, 1)
        row_temp = W(i, :); % 行向量
        row_vector = row_temp'; 
        part2 = part2 + lambda1 * norm(row_vector, 1) ...
                      + lambda2 * norm(row_vector, 2);
    end  
    

    part3 = 0;
    fused_W = W * H;
    for i = 1 : size(fused_W, 2)
        col_temp = fused_W(:, i);
        part3 = part3 + lambda3 * norm(col_temp, 1);
    end
    
    val = part1 + part2 + part3;
    

    
end




function Out = prox_SGL(V ,lambda1, lambda2)
    Out = zeros(size(V));

    for row = 1 : size(V, 1)

        target = V(row, :); % 目标行向量
        t_Lasso = max(abs(target) - lambda1, 0) .* sign(target);
             
        
        out_temp = norm(t_Lasso);  % 2范数
        if out_temp  == 0
            out = 0;
        else
            out = max(out_temp - lambda2, 0) / out_temp * t_Lasso;
        end
        Out(row, :) = out;
    end
end


%% 列分解求邻近算子
% 1/2||W-V||_F^2 + \lambda_3||W||_1
function Out = prox_FL(V, lambda)
    Out = zeros(size(V));
  
    for col = 1 : size(V, 2)
        target = V(:, col); % 目标列向量
        out_temp = max(abs(target) - lambda, 0); % 全域收缩
        out = sign(target) .* out_temp; % 填上符号
        Out(:, col) = out;
    end
end




function [ funcVal ] = unit_funcVal_eval( w, x, y)
%function value evaluation for each task
m = length(y);
%  m = 1;
weight = ones(m, 1)/m;
% weight = ones(m, 1);
aa = -y.*(x*w);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end









