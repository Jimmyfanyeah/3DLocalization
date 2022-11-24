% TODO: generate norm_ai used in CEL0 loss (python)

%
global Np L Nzones
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L


%% PART I Generate 3D PSF matrix
num_slice = 250;  % 127;
zeta = linspace(-Nzones*3+1,Nzones*3-1,num_slice);
A = zeros(96,96,num_slice); [Nx,Ny,Nz] = size(A);

Np = Nx;
for k = 1 : num_slice
    V = [0,0, zeta(k) 2000];
    I0 = RPSFSpiralPhase(Nzones,Np,L,zeta(k));
    A(:,:,end-k+1) = I0/norm(I0(:));
end

%% visualization of A
% A_plot = find(A>0.04);
% px = [];
% py = [];
% pz = [];
% for n = 1 : numel(A_plot)
% [p_x, p_y, p_z] = ind2sub(size(A),A_plot(n));
% px = [px p_x];
% py = [py p_y];
% pz = [pz p_z];
% end
% figure;
% c = linspace(1,10,length(pz));
% h = scatter3(px,py,pz,[],c,'filled');

%% PART II Generate norm_ai
norm_ai = zeros(size(A));
for u=1:96
    for v=1:96
        tic
        for w=1:num_slice
%             tic
            fprintf('%02d-%02d-%03d',u,v,w);
            delta = zeros(size(A));
            delta(u,v,w) = 1;
            a3d = convn(A,delta,'same');
            a2d = a3d(:,:,floor(num_slice/2)+1);
%             figure(1); imshow(a2d,[]);
            norm_ai(u,v,w) = norm(a2d,"fro");
%             toc
            fprintf('\b\b\b\b\b\b\b\b\b')
        end
        toc
        save('./norm_ai_D250.mat','norm_ai')
        fprintf('save %02d %02d\n',u,v)
    end
end

save('./norm_ai_D250.mat','norm_ai')