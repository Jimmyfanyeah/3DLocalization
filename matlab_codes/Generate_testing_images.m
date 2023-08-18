%% Generate train images
% clear; close all
addpath("demo_rpsf")
global   Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L
Np = 96;

% Path and parameters
nSource = 5;
testsize = 100;
Ntest  = [1,testsize];

save_path = ['../../data_test/test',num2str(nSource)];
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

% store information
all_nSource = [];
all_overlap = [];
all_photon = [];
all_flux = [];
all_depth = [];

%% generate images
label_file = fopen([save_path,'/label.txt'],'w');
for ii = Ntest(1):Ntest(2)
    % rng('shuffle');
    rng(ii);


    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true = 2*20*(rand(1,nSource)-0.5);

    
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
    save([save_path,'/im',num2str(ii),'.mat'],'g');
    save([save_path,'/I',num2str(ii),'.mat'],'I0');
    disp([num2str(ii),' saved']);
    % save label
    LABEL = [ii*ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);

end
fclose(label_file);

%% save information
save([save_path,'/photons.mat'],'all_photon');
save([save_path,'/flux.mat'],'all_flux');
save([save_path,'/nSource.mat'],'all_nSource');
save([save_path,'/depth.mat'],'all_depth');




