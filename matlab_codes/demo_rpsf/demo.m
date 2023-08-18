% clear; close all
load('data_natural_order_A'); % Single role
global   Np nSource L Nzones
L = 4; Nzones = 7;
[Nx,Ny,Nz] = size(A); Np = Nx;

N_test  = 100;
datestring = datestr(now,'yyyymmdd');
save_path = ['../../../data_test_var_',datestring,'/test',num2str(nSource)]; % path to save train and noiseless images
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

all_nSource = [];
all_overlap = [];
all_photon = [];
all_flux = [];
all_depth = [];

interest_reg = zeros(32,nSource); 
time = zeros(N_test,1); recall = zeros(N_test,1); 
precision = zeros(N_test,1); ab_error = time;
time_p = zeros(N_test,1); recall_p = zeros(N_test,1);
precision_p = zeros(N_test,1); ab_error_p = ab_error;
flux_total = [];

%================ choose different method =====================
method = 2; % input('Enter a number from 1 to 4: '); 
% 1: KL-L1
% 2: KL-nonconvex
% 3: L2-L1
% 4: L2-nonconvex
%==============================================================

%================ whether with or without flux refinement ========
Flux_ref = 0; 
% 1 is with
% 0 is without
%==================================================================

label_file = fopen([save_path,'/label.txt'],'w');
for nt = 1: N_test
    fprintf('Test %d\n',nt)
    % rng('shuffle');
    rng(nt)

    %% ground true and observed image not on grid point
    Flux_true = poissrnd(2000,[1,nSource]);
    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true =2*20*(rand(1,nSource)-0.5);
    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(nSource,Vtrue); % flux value in normalized basis case

    % Region of interest
    for i = 1 : nSource
        x0 = zeros(size(A));
        xlow = floor(49+Vtrue(i)); 
        ylow = floor(49+Vtrue(i+ nSource));
        zlow = floor((Vtrue(i+2*nSource)+21)/2.1)+1;
        x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*nSource); % 
        interest_reg(:,i) = find(x0~=0);
    end
    b = 10; g = poissrnd(I0+b); % Obversed image
 
    % save mat file
    save([save_path,'/im',num2str(nt),'.mat'],'g');
    save([save_path,'/I',num2str(nt),'.mat'],'I0');
    disp([num2str(nt),' saved']);
    % save labels
    LABEL = [nt*ones(1,nSource); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);

%% Algorithm on localization
    tic
    switch method 
        case 1
    %% KL-L1
        Alg = 'KL-L1';
        if nSource == 15
            p1 = 0.005; p2 =  750; p3 = 0.15;%  15 point sources case
        elseif nSource == 20  
            p1 = 0.010; p2 =  500; p3 = 0.1000;
        elseif nSource == 30
            p1 = 0.010; p2 =  750; p3 = 0.05;
        elseif nSource == 40
            p1 = 0.010; p2 =  300; p3 = 0.1000;
        else
             p1 =.01; p2 = 750; p3= 0.1;
        end
        [u1] = ADMM_poisson_neg_steplength(g, A,5,p1,p2,p3); % for kl-l1 nonnegative 

        case 2
    %% IRL1-nonconvex
        Alg = 'KL-NC';
        if nSource == 30 || nSource == 35
            mu = 0.0100;a = 50.0000; nu = 0.0150; lambda = 40.0000; 
        elseif nSource == 40 || nSource == 45
            mu = 0.005;a = 50.0000; nu = 0.0250; lambda = 40.0000;
        elseif nSource == 15 || nSource == 20 || nSource == 25
            mu = 0.0015;a = 80.0000; nu = 0.0300;lambda = 120.0000;
        else
           mu = 0.0011;a = 80.0000; nu = 0.0400;lambda = 140.0000; % 5 & 10
        end
        u1 = IRL1_poisson_steplength(g,A,b,a, mu,nu,lambda);
        case 3
    %% L2-L1
        Alg = 'L2-L1';
        if nSource == 15
            p1 = 0.150; p2 =  500; p3 = 0.1;%  15 point sources case
        elseif nSource == 20  || nSource == 30
            p1 = 0.100; p2 =  800; p3 = 0.100; % 20 point sources case 
        elseif nSource == 40
            p1 = 0.1500; p2 =  600; p3 = 0.100; 
        else
           p1 = 0.050; p2 =  850;p3 = 0.400; % 5, 10 point sources case
        end
        [u1] = ADMM_l2_l1_steplength(g-5, A,p1,p2,p3);    
        case 4  
    %% L2-nonconvex
        Alg = 'L2-NC';
        if nSource == 5 || nSource == 10 
            mu = 0.0005;a = 20.0000; nu = 0.0700; lambda = 160.0000;
        elseif nSource == 15
            mu = 0.0005;a = 20.0000; nu = 0.0700; lambda = 200.0000;
        elseif nSource == 40
             mu = 0.0250;a = 20.0000; nu = 0.1500; lambda = 200.0000;
        elseif nSource == 30
            mu = 0.0005;a = 20.0000; nu = 0.2500; lambda = 160.0000;
        else
            mu = 0.0005;a = 20.0000; nu = 0.150;lambda = 160.0000; % 20
        end
        [u1] = IRL1_l2_steplength(g-5,A,b,a, mu,nu,lambda);    

    end

%% Removing the clustered false positive 
   time(nt) = toc;
   [xIt, elx, ely, elz] = local_3Dmax_large(u1);
    
%% Iterative Scheme on refinment on estimation of flux & Evaluation
    if Flux_ref == 1
        idx_est = find(xIt>0); 
        [flux_new] = Iter_flux(A, idx_est, g, b);
        [re, pr,flux_tem, flux_est] = ...
            Eval_v2(xIt, interest_reg,flux_new,flux); 
    else 
        [re, pr] = Eval_v1(xIt, interest_reg); % without flux refinement
    end

    recall(nt) = re;
    precision(nt) = pr;
    if recall(nt) == 0
        break;
    end

    [re_p, pr_p] = Eval_v1(u1, interest_reg); 
    recall_p(nt) = re_p;
    precision_p(nt) = pr_p;

    fprintf('%s in %d point source case\n',Alg,nSource);
    fprintf('Recall = %3.2f%%\n',recall(nt)*100);
    fprintf('Precision = %3.2f%%\n',precision(nt)*100);
    fprintf('Cost time = %3.2f seconds\n',time(nt));
    fprintf('---\n');
 
end
fclose(label_file);

mean_recall=mean(recall);
mean_precision=mean(precision);
mean_time=mean(time);
dlmwrite('../../../test_output/var/result_var.csv',{N_test,nSource,mean_precision,mean_recall,mean_time},'delimiter',',','-append');