function [recall, precision,jaccard_index, flux_total, flux_est, tp_pred, tp_gt] = Eval_v2(xIt, interest_reg,flux_new,flux)
% xIt: 3d grid with nonzero entries are predicted pts
% flux_new: estimated flux values based on predicted 3d location
% flux: ground truth flux values
global nSource
idx_est = find(xIt>0); 
x_est = zeros(size(xIt));

x_est(idx_est) = 1;
num_nonz = length(flux);
num_est = numel(find(x_est~=0));
num_tr = 0;
flux_est = [];flux_total = [];
xIt_dnn = xIt;
xIt(idx_est) = flux_new(:,2);
% xIt_dnn(idx_est) = flux_new(:,2);

tp_pred = []; tp_gt = [];
for i = 1:length(flux)
    tem = intersect(interest_reg(:,i),idx_est);
    if numel(tem)~=0
        num_tr = num_tr + 1;
        idx_con = idx_est(idx_est==tem(1)); idx_con = idx_con(1);
        tp_pred = [tp_pred, idx_con]; tp_gt = [tp_gt,i];
        flux_add = [flux(i); xIt(idx_con); xIt_dnn(idx_con)];
        flux_est = [flux_est flux_add];
        idx_est(idx_est==tem(1)) = [];
    end
end
flux_total = [flux_total flux_est];
% Add the fasle postive in the estimated result flux_est 
for i = 1:numel(idx_est)
    flux_add = [0; xIt(idx_est(i)); xIt_dnn(idx_est(i))];
    flux_est = [flux_est flux_add];
end
recall =  num_tr/num_nonz;
precision = num_tr/num_est; 
jaccard_index = num_tr/(num_nonz+num_est-num_tr);

fprintf('TP = %d, P = %d, Target = %d\n',num_tr,num_est,num_nonz);
end

