function sol=  IRL1_poisson_steplength(g,A,b,a, mu,nu,lambda)
% IRL1 for KL-NC model


Nout_iter = 2;
Nin_iter = 400;
y = g(:)';
[ny, nx, nz] = size(A);
T = zeros(1, nz); T(nz) = 1;
fx = zeros(size(A));
feta0 = fx; feta1 = fx;
fA = fftn(A);
invall = abs(fA).^2+nu/mu;
u0_2d = T'*(-ones(1,ny*nx)*(1+mu*b)+sqrt(ones(1,ny*nx)*(1-mu*b)^2+4*mu*y))/(2*mu);
u0 = reshape(u0_2d', ny, nx, nz);
fu0 = fftn(u0);
u1 = zeros(size(A));
fu1 = fftn(u1);

N_eta0 = zeros(2,Nin_iter);
N_eta1 = N_eta0;
rel_error = N_eta0;
num_pos = N_eta0;
sl = 1.618;

for oi = 1 : Nout_iter
    

    absx = abs(u1);
    weights = a*lambda/((a+absx).^2);
    for ii = 1:Nin_iter
        uold = u1;
        ftemp_1 = conj(fA).*(fu0-feta0);
        fx = (ftemp_1+nu/mu*(fu1-feta1))./invall;% update x, equation (7)
        fAx = fA.*fx;
        feta0 = feta0-sl*(fu0-fAx);% update eta0, equation (8)
        feta1 = feta1-sl*(fu1-fx);% update eta1, equation (9)
        
        % checking convergence speed
%         eta0 = ifftn(fu0-fAx);
%         N_eta0(oi,ii) = norm(eta0(:)); 
%         eta1 = ifftn(fu1-fx);
%         N_eta1(oi,ii) = norm(eta1(:)); 
%         
        
        Ax_eta_2d = ifftn(fAx+feta0);
        if nz > 1 
            Ax_eta_2d = reshape(XYZ_rot(Ax_eta_2d, [3,1,2]), nz, ny*nx);
        else
            Ax_eta_2d = Ax_eta_2d(:)';
        end
        TT1 =  mu* Ax_eta_2d(end,:)- (1+mu*b);       
        u0_2d = Ax_eta_2d;
        u0_2d(end,:) = (TT1 + sqrt(TT1.^2+4*mu*(y-b+b*mu*Ax_eta_2d(end,:))))./(2*mu);
        u0 = reshape(u0_2d', ny, nx, nz);% update u0, equation (5)
        fu0 = fftn(u0);
        x_eta1 = ifftn(fx+feta1);
        num_pos(oi,ii) = numel(find(x_eta1-weights./nu <0));
        u1 = max(x_eta1-weights./nu, 0);% update u1, equation (6) % for ADMM_deconv.m
        fu1 = fftn(u1);
        
        rel_error(oi,ii) = norm(u1(:)-uold(:))/(norm(uold(:))+eps);

%         if ii/50 == round(ii/50)
%         fprintf('Rel_error: %2.1e, Outer: %2d, Inner: %2d; \n',...
%             rel_error(oi,ii),oi, ii);
%         end

    end   

        if rel_error < 1e-5
            break;
        end           

end


% figure; 
% subplot(2,2,1)
% plot(N_eta0(1,:));
% title('1st outer iteration,  ||U_0-A*X||')
% subplot(2,2,2)
% plot(N_eta0(2,:));
% title('2nd outer iteration,  ||U_0-A*X||')
% subplot(2,2,3)
% plot(N_eta1(1,:));
% title('1st outer iteration, ||U_1-X||')
% subplot(2,2,4)
% plot(N_eta1(2,:));
% title('2nd outer iteration, ||U_1-X||')
% 
% figure;
% subplot(2,1,1)
% plot(250:400, rel_error(1,250:400));
% title('1st outer iteration, error')
% subplot(2,1,2)
% plot(100:400, rel_error(2,100:400));
% title('2nd outer iteration, error')

% figure; 
% subplot(2,1,1)
% plot(num_pos(1,:));
% title('1st outer iteration, number of negative entires')
% subplot(2,1,2)
% plot(num_pos(2,:));
% title('2nd outer iteration, number of negative entires')



u1 = ifftshift(u1);
u1 = fftshift(u1,3);
sol = u1;