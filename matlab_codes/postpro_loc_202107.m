%% post-processing
load('data_natural_order_A'); % Single role 
global   Np nSource L Nzones
L = 4; Nzones = 7;
[Nx,Ny,Nz] = size(A); Np = Nx;
N_test  = 50;
b = 5;
nSource = 25;
% fixed_flux_value = 110;

% pred_path = folder save loc.csv
pred_path = ['/home/tonielook/Documents/3dloc/Result/202107/test_result'];


pred = readtable([pred_path,'/loc.csv']);
pred = table2array(pred);
gt = readtable([pred_path,'/label.txt']);
gt = table2array(gt(:,1:5));

recall = zeros(50,1);
precision = zeros(50,1);
new_nSources = zeros(50,1);

flux_all = [];
for nt = 1:50
    gt_tmp = gt(gt(:,1)==nt,:);
    gt_trans = [gt_tmp(:,1),gt_tmp(:,2)+49,gt_tmp(:,3)+49,gt_tmp(:,4)+20,gt_tmp(:,5)];
    % add flux threshold in gt
%     gt_tmp = gt_tmp(gt_tmp(:,5)>fixed_flux_value,:);

    pred_tmp = pred(pred(:,1)==nt,:);

    new_nSource = length(gt_tmp);
    new_nSources(nt) = new_nSource;
    
    interest_reg = zeros(32,new_nSource); 
    % Region of interest == ground-truth
    Vtrue = [gt_tmp(:,3);gt_tmp(:,2);gt_tmp(:,4);gt_tmp(:,5)];
    flux = gt_tmp(:,5);
    for i = 1 : new_nSource
        x0 = zeros(size(A));
        xlow = floor(49+Vtrue(i)); 
        ylow = floor(49+Vtrue(i+ new_nSource));
        zlow = floor((Vtrue(i+2*new_nSource)+21)/2.1)+1;
        x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*new_nSource); % 
        interest_reg(:,i) = find(x0~=0);
    end
    
    % prediction
    Vpred = [pred_tmp(:,3);pred_tmp(:,2);pred_tmp(:,4);pred_tmp(:,5)];
    pred_bol = zeros(size(A));
    nPred = length(Vpred)/4;
    for i = 1 : nPred
        xlow = round(49+Vpred(i)); 
        ylow = round(49+Vpred(i+ nPred));
        if xlow>96
            xlow = 96;
        end
        if ylow>96
            ylow=96;
        end
        zlow = round((Vpred(i+2*nPred)+21)/2.1)+1;
        pred_bol(xlow,ylow,zlow)= pred_bol(xlow,ylow,zlow)+Vpred(i+3*nPred);
    end

    % Removing the clustered false positive 
    [xIt, elx, ely, elz] = local_3Dmax_large(pred_bol);
    
    % Iterative Scheme on refinment on estimation of flux & Evaluation
    idx_est = find(xIt>0); 
    flux_nn = xIt(idx_est);
    if isempty(idx_est)
        continue
    end
    [flux_new] = Iter_flux(A, idx_est, g, b);

    % Evaluation
    [re, pr,flux_total, flux_est] = Eval_v2(xIt, interest_reg,flux_new,flux); 
    % [re, pr] = Eval_v1(xIt, interest_reg); % for process without flux refinement
    recall(nt) = re;
    precision(nt) = pr;

    fprintf('%d\n',nt)
    fprintf('%d point source case\n',new_nSource);
    fprintf('Recall = %3.2f%%\n',recall(nt)*100);
    fprintf('Precision = %3.2f%%\n',precision(nt)*100);
    fprintf('---\n');
    
    flux_all = [flux_all,flux_total];
    
    % Plot flux estimation
%     figure(1); 
%     w1 = 0.5; w2 = 0.25;
%     bar(flux_est(1,:),'FaceColor',[0.2 0.2 0.5])
%     hold on 
%     bar(flux_est(2,:),w2,'FaceColor',[0 0.7 0.7])
%     legend('true','est')
%     title(num2str(nt));
%     set(gcf,'position',[100,100,1200,600]);
%     saveas(gcf,[histogram_save_path,'\',num2str(nt),'.fig']);
%     saveas(gcf,[histogram_save_path,'\',num2str(nt),'.png']);
%     hold off

end


%% plot the 3D estimation (compare ground true with estimated solution)
[loc_x,loc_y,loc_z] = ind2sub(size(A),find(xIt>0)); 
[locx_v0,locy_v0,locz_v0] = ind2sub(size(A),find(pred_bol>0));

Ax_eta_2d=ifftn(fftn(A).*fftn(ifftshift(fftshift(xIt),3))); 
J = Ax_eta_2d(:,:,Nz);
figure;
scatter3(Vtrue(1:new_nSource)+49,Vtrue(new_nSource+1:2*new_nSource)+49, ...
    (Vtrue(2*new_nSource+1:3*new_nSource)+21)/2.1+1,'ro')
axis([1 96 1 96 -0 21])
% axis([-48 48 -48 48 0 21])
hold on
scatter3(loc_x,loc_y,loc_z,'b+')
hold on
legend('true','est','Location','Southoutside','Orientation','horizontal')
title(['img',num2str(nt)])

%% plot flux estimation
% figure; % Plot flux estimation
% w1 = 0.5; w2 = 0.25;
% bar(flux_est(1,:),'FaceColor',[0.2 0.2 0.5])
% hold on 
% bar(flux_est(2,:),w2,'FaceColor',[0 0.7 0.7])
% legend('true','est')

% flux_per = abs(flux_all(1,:)-flux_all(2,:))./flux_all(1,:);
% figure; % plot the relative error in flux wrt histogram 
% histogram(flux_per)
% title(num2str(nSource));
