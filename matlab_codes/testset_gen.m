
global Np nSource L Nzones
L = 4; Nzones = 7; 

% INPUT
% nSource = 15;
% N_test  = 100;
% save_path = '';
% noise_type = "poisson"; % poisson

nSource = str2num(nSource);
N_test = str2num(N_test);

Np = 96;
interest_reg = zeros(32,nSource);


% Path

train_path = fullfile(save_path,'noise');  % path to save train images with noise
clean_path = fullfile(save_path,'clean');  % path to save noiseless ground-truth images
if ~exist(train_path, 'dir') || ~exist(clean_path, 'dir')
   mkdir(train_path)
   mkdir(clean_path)
end

label_path = fullfile(save_path,sprintf('label.txt',nSource));
label_file = fopen(label_path,'w');

for nt = 1:N_test
    fprintf('Test %03d/%03d...',nt,N_test);
    rng(150*nt)
%% ground true and observed image not on grid point
    real_pos = zeros(nSource, 3);
    %%-------------- small region--------------------
    Flux_true = poissrnd(2000,[1,nSource]);
    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true = 2*20*(rand(1,nSource)-0.5);
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case
    %[I0] = PointSources_poisson(nSource,Vtrue); flux = Flux_true;

    % b = 5; g = poissrnd(I0+b); % Obversed image
    b = 5;
    % obversed image with poisson noise
    if noise_type == "gaussian"
        sigma = 0.1;
        g = I0 + b + sigma*randn(Np);
    elseif noise_type == "poisson"
        g = poissrnd(I0+b);
    end

    % Save mat files
    save([train_path,'/im',num2str(nt),'.mat'],'g');
    save([clean_path,'/im',num2str(nt),'.mat'],'I0');
    % Save Label
    LABEL = [nt*ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f\n',LABEL);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
end
fclose(label_file);


