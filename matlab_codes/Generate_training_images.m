%% Generate train images
% clear; close all
addpath("demo_rpsf")
global   Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L
Np = 96;

% Path and parameters
nSource = 70;
Ntest  = [1,10000];
base_path = '../../data_train/';
train_path = [base_path,'train/'];  % path to save train images with noise
clean_path = [base_path,'clean/']; % path to save noiseless ground truth images
if ~exist(train_path, 'dir') || ~exist(clean_path, 'dir')
   mkdir(train_path)
   mkdir(clean_path)
end

% store information
all_nSource = [];
all_overlap = [];
all_photon = [];
all_flux = [];
all_depth = [];

%% generate images
datestring = datestr(now,'yymmddHH');
label_file = fopen([train_path,'label.txt'],'w');
for ii = Ntest(1):Ntest(2)
    % rng('shuffle');
    rng(ii);

    % add additional overlap point sources
    overlap_pts = min(randi([1,4]),floor(nSource/2));
    all_overlap = [all_overlap,overlap_pts];

    Xp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    Yp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    zeta_true = 2*20*(rand(1,nSource-overlap_pts)-0.5);

    % add extra overlap points
    for jj=1:length(Xp_true)
        while length(Xp_true) < nSource
            a = Xp_true(jj)+(rand(1)-0.5)*4;
            b = Yp_true(jj)+(rand(1)-0.5)*4;
            c = zeta_true(jj)+(rand(1)-0.5)*5;
            if abs(a)<34 && abs(b)<34 && abs(c)<20 && 2<abs(zeta_true(jj)-c)
                Xp_true = [Xp_true,a];
                Yp_true = [Yp_true,b];
                zeta_true = [zeta_true,c];
            end
        end
    end

    Flux_true = poissrnd(2000,[1,length(Xp_true)]); % Random photon numbers from Poisson distribution

    %% generate image based on 3d location and photon values
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case
%     figure; imshow(I0,[])

    b = 10; g = poissrnd(I0+b); % Obversed image with poisson noise
%     figure;imshow(g,[])

    all_photon = [all_photon;Flux_true'];
    all_flux = [all_flux; flux];
    all_depth = [all_depth;zeta_true'];

    % save mat file
    save([train_path,'im',num2str(ii),'.mat'],'g');
    save([clean_path,'I',num2str(ii),'.mat'],'I0');
    disp([num2str(ii),' saved']);
    % save label
    LABEL = [ii*ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);

end
fclose(label_file);
copyfile '../../data_train/train/label.txt' '../../data_train/train/label.txt.bak';

%% save information
save([base_path,'photons.mat'],'all_photon');
save([base_path,'flux.mat'],'all_flux');
save([base_path,'nSource.mat'],'all_nSource');[0]
save([base_path,'depth.mat'],'all_depth');
save([base_path,'overlap.mat'],'all_overlap_pts');

