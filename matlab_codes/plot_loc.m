%% plot 3d points of prediction and ground truth in loc.csv size=96*96*40

nSource = 5;
pred_path = '  ';
pred_path = [pred_path, '\test',num2str(nSource)];

gt = readtable([pred_path,'\label.txt']);
gt = table2array(gt(:,1:5));
pred = readtable([pred_path,'\loc.csv']);
pred = table2array(pred);

% Ntest = length(gt)/nSource;
for nt = 1

pred_tmp = pred(pred(:,1)==nt,:);
gt_tmp = gt(gt(:,1)==nt,:);

%% remove boundary artifact
% pred_tmp_after = pred_tmp(abs(pred_tmp(:,2))~=47.75,:);
% pred_tmp_after = pred_tmp_after(abs(pred_tmp_after(:,3))~=47.75,:);
% pred_tmp = pred_tmp(abs(pred_tmp(:,4))~=19.914,:);

%% plot graph
figure;
plot3(gt_tmp(:,3),gt_tmp(:,2),gt_tmp(:,4),'ro',pred_tmp(:,3),pred_tmp(:,2),pred_tmp(:,4),'bx');

axis([-49 49 -49 49 -20 20])
title(['img',num2str(nt)])
grid on

end

%% load original image
% mat_path = ['F:\3d_localization\data\test\test_v3_1st\test',num2str(nSource)];
% load([mat_path,'\I',num2str(id),'.mat']);  % mat file for I0
% figure; imagesc(I0.')



