%% Generate train images
% clear; close all
global   Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L
Np = 96;

%% Modify path and params
Ntest  = [0,1];

base_path = ''; % save path
train_path = [base_path,'\train\'];  % path to save train images with noise
clean_path = [base_path,'\clean\']; % path to save noiseless ground truth images
if ~exist(train_path, 'dir') || ~exist(clean_path, 'dir')
   mkdir(train_path)
   mkdir(clean_path)
end

% nSource is a random value uniformly distributed in [1,40]
rng(1024);
all_nSource = randi([1,40],[Ntest(2),1]);
all_photon = [];
all_flux = [];
all_depth = [];
all_overlap = [];

%% generate images
label_file = fopen([train_path,'\label.txt'],'w');
for ii = Ntest(1):Ntest(2)-1
    rng(ii);
    nSource = all_nSource(ii+1);
    overlap_pts = min(randi([1,4]),floor(nSource/2));
    all_overlap = [all_overlap,overlap_pts];

    Xp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    Yp_true = 34*2*(rand(1,nSource-overlap_pts)-0.5);
    zeta_true = 2*20*(rand(1,nSource-overlap_pts)-0.5);
    
    %% add extra overlap points
    for jj=1:length(Xp_true)
        while length(Xp_true) < nSource
            a = Xp_true(jj)+(rand(1)-0.5)*4;
            b = Yp_true(jj)+(rand(1)-0.5)*4;
            c = zeta_true(jj)+(rand(1)-0.5)*3;
            if abs(a)<34 && abs(b)<34 && abs(c)<20 && 2<abs(zeta_true(jj)-c)
                Xp_true = [Xp_true,a];
                Yp_true = [Yp_true,b];
                zeta_true = [zeta_true,c];
            end
        end
    end
    
    %% flux
%     Flux_true = poissrnd(2000,[1,length(Xp_true)]); % Random numbers from Poisson distribution
%     Flux_true = ones(1,length(Xp_true))*2000;  % Photon = 2000

    % flux follow uniform distribution
    p = [3.0494e-05, -2.8545e-04, 0.0210, 0.0069, 13.3277];
    pred_ratio = p(1)*abs(zeta_true').^4 + p(2)*abs(zeta_true').^3 + p(3)*abs(zeta_true').^2 + p(4)*abs(zeta_true') + p(5);
    Flux_true = (rand(1,nSource)-0.5)*2*50+120;
    Flux_true = pred_ratio'.*Flux_true;
    
    %% generate image based on 3d location and photon values
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case
%     figure; imshow(I0,[])

    b = 5; g = poissrnd(I0+b); % Obversed image with poisson noise
%     figure;imshow(g,[])

    all_photon = [all_photon;Flux_true'];
    all_flux = [all_flux; flux];
    all_depth = [all_depth;zeta_true'];

    % save mat file
    save([train_path,'\im',num2str(ii),'.mat'],'g');
    save([clean_path,'\I',num2str(ii),'.mat'],'I0');
    disp([num2str(ii),' saved']);
    % save labels
    LABEL = [ii*ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);

end
fclose(label_file);

%% save all_flux, all_nSource, all_nSource_v1
save([base_path,'photons.mat'],'all_photon');
save([base_path,'flux.mat'],'all_flux');
save([base_path,'nSource.mat'],'all_nSource');
save([base_path,'depth.mat'],'all_depth');
save([base_path,'overlap.mat'],'all_overlap');

%% visualize flux, all_nSource, all_nSource_v1
figure; histogram(all_photon); title('Flux: photon');
figure; histogram(all_flux); title('flux');
figure; histogram(all_depth); title('depth');
figure; histogram(all_nSource); title('nSource');
figure; histogram(all_overlap); title('num of overlap points');

%% visualize 3d plot
% figure;
% plot3(Xp_true,Yp_true,zeta_true,'o');
% xlabel('y')
% ylabel('x')
% zlabel('zeta')
% xlim([-48 48])
% ylim([-48 48])
% zlim([-20 20])
% ax = gca;
% ax.YDir = 'reverse';
% grid on




