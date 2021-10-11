function [recall, precision, flux_total, flux_est, x, y, z] = Eval_v2(xIt, interest_reg,flux_new,flux)

global nSource
idx_est = find(xIt>0); 
x_est = zeros(size(xIt));

x_est(idx_est) = 1;
num_nonz = length(flux);
num_est = numel(find(x_est~=0));
num_tr = 0;
flux_est = [];flux_total = [];
xIt(idx_est) = flux_new;
loc = [];
for i = 1:length(flux)
    tem = intersect(interest_reg(:,i),idx_est);
    if numel(tem)~=0
        num_tr = num_tr + 1;
        idx_con = idx_est(idx_est==tem(1)); idx_con = idx_con(1);
        loc = [loc, idx_con];
        flux_add = [flux(i); xIt(idx_con)];
        flux_est = [flux_est flux_add];
        idx_est(idx_est==tem(1)) = [];
    end
end
flux_total = [flux_total flux_est];
% Add the fasle postive in the estimated result flux_est 
for i = 1:numel(idx_est)
    flux_add = [0; xIt(idx_est(i))];
    flux_est = [flux_est flux_add];
end
recall =  num_tr/num_nonz;
precision = num_tr/num_est; 

fprintf('TP = %d, P = %d, Target = %d\n',num_tr,num_est,num_nonz);
end

