%% Generate training dataset
% clear; close all

%%%%%%% Input params:
% Nindex_end: the last index of samples, start from 1
% noise_type: poisson & gaussian
% base_path: save path of images and labels
% nS: number of point source, nS=0 <=> random in [3,50]

% Nindex_end = 10;
% base_path = '/home/lingjia/Documents/rpsf/20221121_unrolling/data/20230223_check_equal_increaseBS_10_pt5L7';
% noise_type = "poisson"; % poisson & gaussian
% nS = 1


%% Params in RPSF
load('data_natural_order_A'); % Single role 
global Np L Nzones nSource
L = 4; Nzones = 7;
[Nx,Ny,Nz] = size(A); Np = Nx;

%% Modify path and param
save_info = 1;
if nS == 0
    num_nSource = randi([3,50],[Nindex_end,1]);
else
    num_nSource = ones([Nindex_end,1])*nS;
end
% num_overlap = min(randi([1,4],[Nindex(2),1]),floor(num_nSource/2));
% num_overlap = randi(1,[Nindex(2),1]);

Nindex  = [1,Nindex_end]; % for train or test
% N_test = Nindex_end;

if save_info
    % Save 2D observed clean and noisy images
    train_path = fullfile(base_path,'noise');
    clean_path = fullfile(base_path,'clean');
    if ~exist(train_path, 'dir') || ~exist(clean_path, 'dir')
        mkdir(train_path)
        mkdir(clean_path)
    end

end


rng(1024);
photon_list = [];
flux_list = [];
zeta_list = [];
x_list = [];
y_list = [];


%% Generate image
if save_info
    label_file = fopen(fullfile(base_path,'label.txt'),'w');
end

for nt = Nindex(1): Nindex(2)
    tic
    fprintf('save %05d/%05d...',nt,Nindex(2));
    rng(nt);
    nSource = num_nSource(nt);
    overlap_pts = 0; % use 1 usually

    % Randomly generate x & y & zeta coordinates
    Xp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    Yp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    zeta_true = 20*2*(rand(1,nSource-overlap_pts)-0.5);

    % Add extra overlap points
%     for jj=1:length(Xp_true)
%         while length(Xp_true) < nSource
%             a = Xp_true(jj)+(rand(1)-0.5)*4;
%             b = Yp_true(jj)+(rand(1)-0.5)*4;
%             c = zeta_true(jj)+(rand(1)-0.5)*35;
%             if abs(a)<34 && abs(b)<34 && abs(c)<20 && 2<abs(zeta_true(jj)-c)
%                 Xp_true = [Xp_true,a];
%                 Yp_true = [Yp_true,b];
%                 zeta_true = [zeta_true,c];
%             end
%         end
%     end

%    % Remove point sources with dist(x,y,z)<2
%     kk=1;
%     while kk <= length(Xp_true)
%         qq = kk + 1;
%         while qq <= length(Xp_true)
%             if abs(Xp_true(kk)-Xp_true(qq))<=2 && abs(Yp_true(kk)-Yp_true(qq))<=2 && abs(zeta_true(kk)-zeta_true(qq))<=2
%                 Xp_true(qq) = [];
%                 Yp_true(qq) = [];
%                 zeta_true(qq) = [];
% %                 fprintf('Overlap case %d and %d\n',kk,qq);
%             else
%                 qq = qq+1;
%             end
%         end
%         kk = kk+1;
%     end

    %% assign photon value
    % CASE 1 - Random numbers from Poisson distribution
    Flux_true = poissrnd(2000,[1,length(Xp_true)]);

    % CASE 2 - Fixed numer of photon = 2000
    % Flux_true = ones(1,length(Xp_true))*2000;

    % CASE 3 - Corresponding photon number such that flux follow uniform distribution
    % p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    % pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    % Flux_true = (rand(1,length(Xp_true))-0.5)*2*50+120;
    % Flux_true = pred_ratio'.*Flux_true;

    %% Generate image based on 3d location and photon values
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(length(Xp_true),Vtrue); % flux value in normalized basis case
    % figure; imshow(I0,[]);

    b = 5;
    % obversed image with poisson noise
    if noise_type == "gaussian"
        sigma = 0.1;
        g = I0 + b + sigma*randn(Np);
    elseif noise_type == "poisson"
        g = poissrnd(I0+b);
    end
    % g = I0;x
    % figure;imshow(g,[])

    % save mat file (images) and labels
    if save_info
        save(fullfile(train_path,['im',num2str(nt),'.mat']),'g');
        save(fullfile(clean_path,['im',num2str(nt),'.mat']),'I0');
        LABEL = [nt*ones(1,length(Xp_true)); Yp_true; Xp_true; zeta_true; flux'];
        fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);
    end

    % save overall info
    photon_list = [photon_list;Flux_true'];
    flux_list = [flux_list; flux];
    zeta_list = [zeta_list;zeta_true'];
    x_list = [x_list; Xp_true'];
    y_list = [y_list; Yp_true'];
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
end
fclose(label_file);

%% Save & Visualize phot, flux, nSource, zeta, xcoords, ycoords
save(fullfile(base_path,'numSource.mat'),'num_nSource');
save(fullfile(base_path,'photon.mat'),'photon_list');
save(fullfile(base_path,'flux.mat'),'flux_list');
save(fullfile(base_path,'zeta.mat'),'zeta_list');
save(fullfile(base_path,'xcoords.mat'),'x_list');
save(fullfile(base_path,'ycoords.mat'),'y_list');

% figure; histogram(num_nSource+1); title('nSource');
% figure; histogram(photon_list); title('Photon');
% figure; histogram(flux_list); title('flux');
% figure; histogram(zeta_list); title('depth');
% figure; histogram(x_list); title('X coords');
% figure; histogram(y_list); title('Y coords');
