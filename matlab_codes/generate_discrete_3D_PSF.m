%% generate A, the 3d discrete PSF matrix
global   Np L Nzones
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L

num_slice = 100;
zeta = linspace(-Nzones*5+1,Nzones*5-1,num_slice);
A = zeros(96,96,num_slice); [Nx,Ny,Nz] = size(A);

Np = Nx;
for k = 1 : num_slice
    V = [0,0, zeta(k) 2000];
    I0 = RPSFSpiralPhase(Nzones,Np,L,zeta(k));
    A(:,:,end-k+1) = I0/norm(I0(:));
end

% plot A, fig 1.1
A_plot = find(A>0.03);
px = [];
py = [];
pz = [];
for n = 1 : numel(A_plot)
[p_x, p_y, p_z] = ind2sub(size(A),A_plot(n));
px = [px p_x];
py = [py p_y];
pz = [pz p_z];
end
figure;
c = linspace(1,10,length(pz));
h = scatter3(px,py,pz,[],c,'filled');