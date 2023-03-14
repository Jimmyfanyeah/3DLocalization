%% Post-processing for initial prediction from NN
%% parameters from generating data
load('data_natural_order_A'); % Single role
global Np nSource L Nzones
L = 4; Nzones = 7; b = 5; [Nx,Ny,Nz] = size(A); Np = Nx;

tic
%% Modify parameters here
save_pred_info = 0; % save pred_label.txt
nSource = 30;

mat_path = [' ',num2str(nSource)]; % path for test data
pred_path = ' '; % path for prediction
pred_path = [pred_path,'\test',num2str(nSource)];
save_path = pred_path;

%% main
pred = readtable([pred_path,'\loc.csv']);
pred = table2array(pred);
gt = readtable([pred_path,'\label.txt']);
gt = table2array(gt(:,1:5));

% evaluation metrics
recall = zeros(50,1);
precision = zeros(50,1);
jaccard_index = zeros(50,1);
f1_score = zeros(50,1);
initial_pred_pts = zeros(50,1);
flux_all = [];

if save_pred_info
    % pred_label: pred 3d locations + flux estimated from var/dnn
    label = fopen([save_path,'\pred_label.txt'],'w');
end

for nt = 1:50
    %% Post-processing
    gt_tmp = gt(gt(:,1)==nt,:);
    pred_tmp = pred(pred(:,1)==nt,:);

    % remove boundary pts
%     pred_tmp_after = pred_tmp(abs(pred_tmp(:,2))~=47.75,:);
%     pred_tmp_after = pred_tmp_after(abs(pred_tmp_after(:,3))~=47.75,:);
%     pred_tmp = pred_tmp(pred_tmp(:,4)~=-19.914,:);

    % load ground truth 3d grid
    interest_reg = zeros(32,nSource); 
    Vtrue = [gt_tmp(:,3);gt_tmp(:,2);gt_tmp(:,4);gt_tmp(:,5)];
    flux_gt = gt_tmp(:,5);
    for i = 1 : nSource
        x0 = zeros(size(A));
        xlow = floor(49+Vtrue(i)); 
        ylow = floor(49+Vtrue(i+nSource));
        zlow = floor((Vtrue(i+2*nSource)+21)/2.1)+1;
        x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*nSource);
        interest_reg(:,i) = find(x0~=0);
    end
    
    % load initial prediction
    Vpred = [pred_tmp(:,3);pred_tmp(:,2);pred_tmp(:,4);pred_tmp(:,5)];
    pred_vol = zeros(size(A));
    nPred = length(Vpred)/4;
    for i = 1 : nPred
        xlow = min(round(49+Vpred(i)),96); 
        ylow = min(round(49+Vpred(i+ nPred)),96);
        zlow = min(round((Vpred(i+2*nPred)+21)/2.1)+1,20);
        pred_vol(xlow,ylow,zlow)= pred_vol(xlow,ylow,zlow)+Vpred(i+3*nPred);
    end

    initial_pred_pts(nt) = numel(find(pred_vol>0));
    % Removing the clustered false positive 
    [xIt, elx, ely, elz] = local_3Dmax_large(pred_vol);
    
    % Iterative Scheme on refinment on estimation of flux & Evaluation
    idx_est = find(xIt>0); 
    if isempty(idx_est)
        continue
    end
    
    flux_est_dnn = xIt(idx_est);
    
    % Estimate flux value
    load([mat_path,'\im',num2str(nt),'.mat']);  % mat file for g
    flux_est_var = Iter_flux(A, idx_est, g, b);

    %% Evaluation
    num_gt = nSource; num_pred = length(idx_est);
    [num_tr,tp_pred,tp_gt,flux_total] = evaluation(xIt, interest_reg, flux_est_var, flux_gt); 

    re =  num_tr/num_gt;
    pr = num_tr/num_pred; 
    ji = num_tr/(num_gt + num_pred - num_tr);
    f1 = 2*(re*pr)/(re+pr);
    
    recall(nt) = re;
    precision(nt) = pr;
    jaccard_index(nt) = ji;
    f1_score(nt) = f1;
   
    fprintf('TP = %d, P = %d, Target = %d\n',num_tr,num_est,num_nonz);    
    fprintf('%d\n',nt)
    fprintf('%d point source case\n',nSource);
    fprintf('Recall = %3.2f%%\n',recall_tmp(nt)*100);
    fprintf('Precision = %3.2f%%\n',precision(nt)*100);
    fprintf('---\n');
    
    %% save pred_label.txt
    if save_pred_info
        [loc_x,loc_y,loc_z] = ind2sub(size(A),find(xIt>0)); 
        LABEL = [nt*ones(1,length(loc_x));loc_y'-49;loc_x'-49; loc_z'*2-21; flux_total(2,:)];
        fprintf(label,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);
    end
end

%% display mean evaluation metrics
mean_precision = sum(precision)/50;
mean_recall = sum(recall)/50;
mean_jaccard = sum(jaccard_index)/50;
mean_f1_score = sum(f1_score)/50;
fprintf('test%d,\nprecision=%.4f, recall=%.4f, jaccard=%.4f, f1 socre=%.4f\n',nSource,mean_precision,mean_recall,mean_jaccard,mean_f1_score);
toc
mean(initial_pred_pts)

%% save info
if save_pred_info
    fclose(label);
    % save precision and recall into csv
    ex = [[1:1:50]',precision,recall,jaccard_index,f1_score];
    writematrix(ex,[save_path,'\eval.csv']);
end

%% plot the 3D estimation (compare ground true with estimated solution)
[loc_x_tp,loc_y_tp,loc_z_tp] = ind2sub(size(A),intersect(tp_pred,find(xIt>0))); 
[loc_x_fp,loc_y_fp,loc_z_fp] = ind2sub(size(A),setxor(tp_pred,find(xIt>0))); 

figure;
% true positive - gt
scatter3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ro')
hold on
% false negative - gt
fn_gt = setxor(1:1:nSource,tp_gt);
scatter3(Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'r^')
hold on
% true positive - pred
scatter3(loc_x_tp,loc_y_tp,loc_z_tp,'bx')
hold on
% false positive - pred
scatter3(loc_x_fp,loc_y_fp,loc_z_fp,'b^')

axis([1 96 1 96 0 21])
legend('tp-gt','fn-gt','tp-p','fp-p','Location','Southoutside','Orientation','horizontal')
% imagesc(I0.')
title(['img',num2str(nt)])

%% save current 3d grid
% filename = [' '_',num2str(nt),'.png']; % save path
% frame = getframe(gca); 
% img = frame2im(frame); 
% imwrite(img,filename); 

%% plot flux estimation for 1 example
% histogram compare gt and prediction
% figure;
% w1 = 0.5; w2 = 0.25;
% bar(flux_total(1,:),'FaceColor',[0.2 0.2 0.5])
% hold on 
% bar(flux_total(2,:),w2,'FaceColor',[0 0.7 0.7])
% legend('true','est')
% title([num2str(nt),' var']);
% set(gcf,'position',[100,100,1200,600]);
% ylim([0,180]);
% % saveas(gcf,[histogram_save_path,'\',num2str(nt),'.fig']);
% % saveas(gcf,[histogram_save_path,'\',num2str(nt),'.png']);
% hold off

% plot the relative error in flux for entire dataset
% figure;
% flux_per = abs(flux_total(1,:)-flux_total(2,:))./flux_total(1,:);
% h1 = histogram(flux_per);
% title(['# of pts ', num2str(nSource)]);
% xlabel('relative error')
% ylabel('# of pts')
