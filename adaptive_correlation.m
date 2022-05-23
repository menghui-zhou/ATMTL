function G = adaptive_correlation(convex, task_num)

    G = eye(task_num);
    for i = 1: task_num-1
        com = eye(task_num);
        com(i, i+1) = convex;
        com(i+1, i+1) = 1-convex;
        G = G * com;
%         disp(G);
    end
    
end

