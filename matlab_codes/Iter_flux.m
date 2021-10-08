function [flux] = Iter_flux(A,idx_est, g, b)
H = [];
% ta_tmp = zeros(96);

for n = 1 : numel(idx_est)
    [p_x, p_y, p_z] = ind2sub(size(A),idx_est(n));
    ta = A(:,:,end+1-p_z);
    point = zeros(size(ta));
    point(p_x,p_y) = 1;
    ta = fftshift(ifft2(fft2(ta).*fft2(point)));
%     ta_tmp = ta_tmp+ta;
    H = [H ta(:)];   % H = PSF matrix
end
% imshow(ta_tmp);
flux_ls = H'*H\(H'*(g(:)-b)); 
flux_old = flux_ls; L_tem = H'*H\H'*(((H*flux_old + b -g(:)).*(H*flux_old))./(H*flux_old + b));
flux_new = flux_ls + L_tem;
error_it = zeros(50,1);
for i = 1 : 48
    L_tem = H'*H\H'*(((H*flux_new + b -g(:)).*(H*flux_new))./(H*flux_new + b));
    flux_tem = flux_ls  + L_tem;
    error_it(i) = norm(flux_new-flux_old)/(norm(flux_old)); 
    flux_old = flux_new;
    flux_new = flux_tem;
end
flux = flux_new;
end