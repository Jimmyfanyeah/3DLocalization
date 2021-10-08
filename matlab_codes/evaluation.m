function [num_tr,tp_pred,tp_gt,flux_total] = evaluation(xIt,interest_reg,flux_est,flux_gt)
% xIt: 3d grid with nonzero entries are predicted pts
% interest_reg: 32 possible position(within standard) for each gt pts, 32-by-nSource
% flux_est: estimated flux values from dnn/var
% flux: ground truth flux values
idx_est = find(xIt>0); 
x_est = zeros(size(xIt));
x_est(idx_est) = 1;

num_nonz = length(flux_gt);
num_est = numel(find(x_est~=0));
num_tr = 0;

flux_total = [];flux_total = [];
xIt(idx_est) = flux_est;
tp_pred = []; tp_gt = [];

for i = 1:num_nonz
    tem = intersect(interest_reg(:,i),idx_est);
    if numel(tem)~=0
        num_tr = num_tr + 1;
        idx_con = idx_est(idx_est==tem(1)); idx_con = idx_con(1);
        tp_pred = [tp_pred, idx_con]; tp_gt = [tp_gt,i];
        flux_add = [flux_gt(i); xIt(idx_con)];
        flux_total = [flux_total flux_add];
        idx_est(idx_est==tem(1)) = [];
    end
end

% Add the fasle postive in the estimated result flux_est 
for i = 1:numel(idx_est)
    flux_add = [0; xIt(idx_est(i))];
    flux_total = [flux_total flux_add];
end

end

