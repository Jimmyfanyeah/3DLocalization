function [recall, precision ] = Eval_v1(u1, interest_reg)

global nSource
idx_est_p = find(u1>0); 
x_est_p = zeros(size(u1));
x_est_p(idx_est_p) = 1;
num_nonz = nSource;
num_est = numel(find(x_est_p~=0));
num_tr = 0;
for i = 1:nSource
    tem = intersect(interest_reg(:,i),idx_est_p );
    if numel(tem)~=0
        num_tr = num_tr + 1;
        idx_est_p(idx_est_p==tem(1)) = [];
    end
end

recall =  num_tr/num_nonz;
precision = num_tr/num_est; 


end
