%% Generate training dataset
% add additional overlap points or not

%%
% clear; close all
global Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 5; % no. of zones in the circular imaging aperture = #L
Np = 96;

%% Modify path and param
Nindex  = [1,10000]; % for Train or Test
save_info = 1;
% base_path = '/Users/dailingjia/Documents/rpsf/gaussian_1k_pt50L5';
% base_path = '/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/data_train/poisson_10k_pt50L5';
% base_path = '/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/data_train/gaussian_10k_pt50L5';
% noise_type = "poisson"; % poisson

train_path = fullfile(base_path,'noise');  % path to save train images with noise
clean_path = fullfile(base_path,'clean'); % path to save noiseless ground-truth images
if ~exist(train_path, 'dir') || ~exist(clean_path, 'dir')
   mkdir(train_path)
   mkdir(clean_path)
end

% when Train
num_nSource = randi([3,50],[Nindex(2),1]);
% num_overlap = min(randi([1,4],[Nindex(2),1]),floor(num_nSource/2));
% num_overlap = randi(1,[Nindex(2),1]);

% when Test
% nSource = 1;
% num_nSource = ones([Ntest(2),1])*nSource;
% num_overlap = zeros([Ntest(2),1]);

rng(1024);
photon_dist = [];
flux_dist = [];
zeta_dist = [];
x_dist = [];
y_dist = [];

%% Generate image
label_file = fopen(fullfile(base_path,'label.txt'),'w');
for ii = Nindex(1): Nindex(2)
    fprintf('save %05d/%05d...',ii,Nindex(2));
    rng(ii);
    nSource = num_nSource(ii);
    overlap_pts = 1;

    % Randomly generate x & y & zeta coordinates
    Xp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    Yp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    zeta_true = 20*2*(rand(1,nSource-overlap_pts)-0.5);

    % Add extra overlap points
    for jj=1:length(Xp_true)
        while length(Xp_true) < nSource
            a = Xp_true(jj)+(rand(1)-0.5)*4;
            b = Yp_true(jj)+(rand(1)-0.5)*4;
            c = zeta_true(jj)+(rand(1)-0.5)*35;
            if abs(a)<34 && abs(b)<34 && abs(c)<20 && 2<abs(zeta_true(jj)-c)
                Xp_true = [Xp_true,a];
                Yp_true = [Yp_true,b];
                zeta_true = [zeta_true,c];
            end
        end
    end

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
%    % CASE 1 - Random numbers from Poisson distribution
%     Flux_true = poissrnd(2000,[1,length(Xp_true)]); 

%    % CASE 2 - Fixed numer of photon = 2000
    % Flux_true = ones(1,length(Xp_true))*2000;

    % CASE 3 - Corresponding photon number such that flux follow uniform distribution
    p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    Flux_true = (rand(1,length(Xp_true))-0.5)*2*50+120;
    Flux_true = pred_ratio'.*Flux_true;

    %% Generate image based on 3d location and photon values
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(length(Xp_true),Vtrue); % flux value in normalized basis case
%     figure; imshow(I0,[]);

    b = 5;
    % obversed image with poisson noise
    if noise_type == "gaussian"
        sigma = 0.1;
        g = I0 + b + sigma*randn(Np);
    elseif noise_type == "poisson"
        g = poissrnd(I0+b);
    end
%     g = I0;x
%     figure;imshow(g,[])

    % save mat file (images) and labels
    if save_info
        save(fullfile(train_path,['im',num2str(ii),'.mat']),'g');
        save(fullfile(clean_path,['im',num2str(ii),'.mat']),'I0');
        LABEL = [ii*ones(1,length(Xp_true)); Yp_true; Xp_true; zeta_true; flux'];
        fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);
    end
    
    % save overall info
    photon_dist = [photon_dist;Flux_true'];
    flux_dist = [flux_dist; flux];
    zeta_dist = [zeta_dist;zeta_true'];
    x_dist = [x_dist; Xp_true'];
    y_dist = [y_dist; Yp_true'];
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
end
fclose(label_file);

%% Save & Visualize phot, flux, nSource, zeta, xcoords, ycoords
save(fullfile(base_path,'numSource.mat'),'num_nSource');
save(fullfile(base_path,'photon.mat'),'photon_dist');
save(fullfile(base_path,'flux.mat'),'flux_dist');
save(fullfile(base_path,'zeta.mat'),'zeta_dist');
save(fullfile(base_path,'xcoords.mat'),'x_dist');
save(fullfile(base_path,'ycoords.mat'),'y_dist');

figure; histogram(num_nSource+1); title('nSource');
figure; histogram(photon_dist); title('Photon');
figure; histogram(flux_dist); title('flux');
figure; histogram(zeta_dist); title('depth');
figure; histogram(x_dist); title('X coords');
figure; histogram(y_dist); title('Y coords');
