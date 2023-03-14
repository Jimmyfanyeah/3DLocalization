% Parameters - demo.py (Generating Data)
load('/Users/dailingjia/Desktop/9000-00-00-00-00-00_test_L7/A.mat'); % Single role
global Np nSource L Nzones
L = 4; Nzones = 7; b = 5; [Nx,Ny,Nz] = size(A); Np = Nx;
zmax = 20;

% Parameters
pred_path_base = '/Users/dailingjia/Desktop/9000-00-00-00-00-00_test_L7';


view = 0; % plot
save_pred_info = 1; % Save pred_coords.csv & eval.csv or NOT

% nsources = [5,10,15,20,30,35,40,50,60];
% nsources = 5:5:45;
nsources = [5];

infer_save = zeros(7,length(nsources));
for nsource_idx = 1:length(nsources)
    nSource = nsources(nsource_idx);

    % Read Ground-truth label_file and Prediction
%     pred_path = fullfile(pred_path_base,"test_"+num2str(nSource));
%     mat_path = fullfile(mat_path_base,['test',num2str(nSource)]);
    pred_path = pred_path_base;
    
    gt = readmatrix(fullfile(pred_path,'label.txt'));
    pred = readmatrix(fullfile(pred_path,'infer_coords.csv'));
    num_imgs = numel(unique(pred(:,1)));

    % start index = 1
%     gt(:,1) = gt(:,1)-min(gt(:,1))+1;
%     pred(:,1) = pred(:,1)-min(pred(:,1))+1;
        
    % Initialize Evaluation Metrics
    recall = zeros();
    precision = zeros();
    jaccard_index = zeros();
    f1_score = zeros();
    initial_pred_pts = zeros();
    final_pred_pts = zeros();
    flux_all = [];

    if save_pred_info
        save_path = pred_path;
   
        label_file = fopen(fullfile(save_path, ['pred_coords_',num2str(nSource),'.csv']), 'w');
        title_label = ["Index", "Int X", "Int Y", "Int Z", "Flux","Float X","Float Y","Float Z","True_Not"];
        fprintf(label_file, [repmat('%s,',1,9),'\n'], title_label);
        
        eval_file = fopen(fullfile(save_path, ['evaluation_',num2str(nSource),'.csv']), 'w');
        title_eval = ["Index","Recall","Precision","Jaccard Index","F1 Score","Initial Pts Num","Final Pts Num"];
        fprintf(eval_file, [repmat('%s,',1,7),'\n'], title_eval);
    end

    tic
    for nt = 1:num_imgs
        gt_temp = gt(gt(:,1)==nt,:);
        pred_temp = pred(pred(:,1)==nt,:);
        
        % load KLNC output
%         load(fullfile('/Users/dailingjia/Desktop/test_L7/raw_output',"u1_final_"+num2str(nt)+".mat"));
%         [u1x, u1y, u1z] = ind2sub(size(A), find(u1>0));
%         u1flux = [];
%         for iidx = 1:length(u1x)
%             u1flux = [u1flux; u1(u1x(iidx),u1y(iidx),u1z(iidx))];
%         end
%         pred_u1 = [u1x,u1y,u1z,u1flux];
%         pred_temp = [ones(length(u1x),1),pred_u1];

        if view
            % View Initial Prediction
            figure(1);
            plot3(gt_temp(:,2)+49,gt_temp(:,3)+49,(gt_temp(:,4)+21)/2.1+1,'ro', ...
            pred_temp(:,3),pred_temp(:,2),pred_temp(:,4),'bx');
            title(sprintf('Image %d Label & Prediction', nt))
            axis([-Np/2 Np/2 -Np/2 Np/2 0 21]); grid on
            
%             plot3(gt_temp(:,3),gt_temp(:,2),(gt_temp(:,4)+21)/2.1,'ro', ...
%             pred_temp(:,2)-48,pred_temp(:,3)-48,pred_temp(:,4)+1,'bx',...
%             pred_u1(:,1)-49, pred_u1(:,2)-49, pred_u1(:,3),'b^');
%             title(sprintf('Label & Prediction & U1 %d', nt))
%             axis([-Np/2 Np/2 -Np/2 Np/2 0 21]); grid on
    %         pause(0.5)
        end

        % Load Ground Truth 3D Grid
        interest_reg = zeros(32,nSource); 
        Vtrue = [gt_temp(:,2);gt_temp(:,3);gt_temp(:,4);gt_temp(:,5)];
        flux_gt = gt_temp(:,5);
        for i = 1 : nSource
            x0 = zeros(size(A));
            xlow = max(floor(49+Vtrue(i)),1);
            ylow = floor(49+Vtrue(i+nSource));
            zlow = floor((Vtrue(i+2*nSource)+21)/2.1)+1;
            x0(xlow-1:xlow+2,ylow-1:ylow+2,zlow:zlow+1)= Vtrue(i+3*nSource);
            interest_reg(:,i) = find(x0~=0);
        end

        % Load Initial Prediction
        Vpred = [pred_temp(:,3);pred_temp(:,2);pred_temp(:,4);pred_temp(:,5)];
        pred_vol = zeros(size(A));
        nPred = length(Vpred)/4;
        for i = 1 : nPred
            xlow = Vpred(i); 
            ylow = Vpred(i+nPred);
            zlow = Vpred(i+2*nPred);
            pred_vol(xlow,ylow,zlow)= pred_vol(xlow,ylow,zlow)+Vpred(i+3*nPred);
        end

        ipts = numel(find(pred_vol>0));
        initial_pred_pts(nt) = ipts;

%         if view
%             % View Initial Prediction to Grid
%             figure(2);
%             [xx,yy,zz] = ind2sub(size(A), find(pred_vol>0));
%             plot3(floor(gt_temp(:,2)), floor(gt_temp(:,3)), floor((gt_temp(:,4)+21)/2.1+1),'ro',...
%                 xx-49, yy-49, zz, 'bx');
%             title(sprintf('Initial Prediction to Grid %d',nt));
%             axis([0 Np+1 0 Np+1 0 21]); grid on;
%             pause(0.5)
%         end

        % Removing Clustered False Positive 
        [xIt, elx, ely, elz] = local_3Dmax_large(pred_vol);
    %     [xIt, elx, ely, elz] = local_3Dmax_large_nm(pred_vol,2,2);

        fpts = numel(find(xIt>0));
        final_pred_pts(nt) = fpts;

        idx_est = find(xIt>0); 
        if isempty(idx_est)
            continue
        end

        flux_est_dnn = xIt(idx_est);
        % Refinment on Estimation of Flux
    %     load(fullfile(mat_path,['im',num2str(nt),'.mat']));  % mat file for g
    %     flux_est_var = Iter_flux(A, idx_est, g, b);

        %% Evaluation
        num_gt = nSource; num_pred = length(idx_est);
        [num_tr,tp_pred,tp_gt,flux_total] = evaluation(xIt, interest_reg, flux_est_dnn, flux_gt);

        re = num_tr/num_gt;
        pr = num_tr/num_pred; 
        ji = num_tr/(num_gt + num_pred - num_tr);
        f1 = 2*(re*pr)/(re+pr);

        recall(nt) = re;
        precision(nt) = pr;
        jaccard_index(nt) = ji;
        f1_score(nt) = f1;

        fprintf('Image %d in %d point source case\n', nt,nSource)
        fprintf('TP = %d, Pred = %d, GT = %d\n',num_tr,num_pred,num_gt);    
        fprintf('Recall = %3.2f%%, Precision = %3.2f%%\n',recall(nt)*100,precision(nt)*100);
        fprintf('---\n');

        %% Save Results
        % TP
        [xxtp,yytp,zztp] = ind2sub(size(A), tp_pred); 
        sxtp = zeros(length(xxtp),1);  sytp = zeros(length(xxtp),1); sztp = zeros(length(xxtp),1);
        retp = ones(length(xxtp),1);
        for sidx = 1: length(xxtp)
            tx = xxtp(sidx); ty = yytp(sidx); tz = zztp(sidx);
            sxtp(sidx) = elx(tx, ty, tz);
            sytp(sidx) = ely(tx, ty, tz);
            sztp(sidx) = elz(tx, ty, tz);
        end

        % FP
        [xxfp,yyfp,zzfp] = ind2sub(size(xIt), setxor(tp_pred, find(xIt>0)));
        sxfp = zeros(length(xxfp), 1);  syfp = zeros(length(xxfp), 1); szfp = zeros(length(xxfp), 1);
        for sidx = 1: length(xxfp)
            tx = xxfp(sidx); ty = yyfp(sidx); tz = zzfp(sidx);
            sxfp(sidx) = elx(tx, ty, tz);
            syfp(sidx) = ely(tx, ty, tz);
            szfp(sidx) = elz(tx, ty, tz);
        end
        refp = zeros(length(xxfp),1);

        xx=[xxtp';xxfp]; yy=[yytp';yyfp]; zz=[zztp';zzfp];
        sx=[sxtp;sxfp]; sy=[sytp;syfp]; sz=[sztp;szfp];
        relist=[retp;refp];

        %% Save pred_label_file.csv & eval.csv
        if save_pred_info
            EST = [nt*ones(1,length(xx))', xx, yy, zz, flux_total(2,:)', sx, sy, sz, relist];
            fprintf(label_file, '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n', EST');

            EVAL = [nt, re, pr, ji, f1, ipts, fpts];
            fprintf(eval_file, '%d,%.4f,%.4f,%.4f,%.4f,%d,%d\n', EVAL);
        end

        %% View
        if view
            % View Final Prediction After Post-pro - Est int
    %         load(fullfile(mat_path,['I',num2str(nt),'.mat']));
            fn_gt = setxor(1:1:nSource,tp_gt);
            figure(3);
            plot3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ro',...
                  Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'r^',...
                  xxtp,yytp,zztp,'bx',...
                  xxfp,yyfp,zzfp,'b^')
            axis([0 96 0 96 0 21]); grid on;
            if isempty(xxfp)
                legend('TP-GT','TP-EST','Location','Southoutside','Orientation','horizontal')
            else
                legend('TP-GT','FN-GT','TP-EST','FP-EST','Location','Southoutside','Orientation','horizontal')
            end
            title(sprintf('Image %d Result after postpro (on grid)',nt))
    %         hold on; imagesc(I0); hold off
            pause(0.5)
% 
%             % View Final Prediction After Post-pro Est float
%             fn_gt = setxor(1:1:nSource,tp_gt);
%             figure(4);
%             plot3(Vtrue(tp_gt)+49,Vtrue(nSource+tp_gt)+49,(Vtrue(2*nSource+tp_gt)+21)/2.1+1,'ro',...
%                   Vtrue(fn_gt)+49,Vtrue(nSource+fn_gt)+49,(Vtrue(2*nSource+fn_gt)+21)/2.1+1,'r^',...
%                   xxtp'+sxtp,yytp'+sytp,zztp'+sztp,'bx',...
%                   xxfp+sxfp,yyfp+syfp,zzfp+szfp,'b^')
%             axis([0 96 0 96 0 21]); grid on;
%             if isempty(xxfp)
%                 legend('TP-GT','TP-EST','Location','Southoutside','Orientation','horizontal')
%             else
%                 legend('TP-GT','FN-GT','TP-EST','FP-EST','Location','Southoutside','Orientation','horizontal')
%             end
%             title(sprintf('Result After Postpro (Est float) %d',nt))
%         %     hold on; imagesc(I0);hold off
%             pause(2)
        end
    end

    %% Display Mean Evaluation Metrics
    mpr = mean(precision);
    mre = mean(recall);
    mjacc = mean(jaccard_index);
    mf1 = mean(f1_score);
    mpts = mean(initial_pred_pts);
    mpts_final = mean(final_pred_pts);
    EVAL = ["Avg", mre, mpr, mjacc, mf1, mpts, mpts_final];
    fprintf(eval_file, '%s,%.4f,%.4f,%.4f,%.4f,%s,%s\n', EVAL);

    fprintf('Total %d Images in %d point source case\n',num_imgs,nSource);
    fprintf('Precision=%.2f%%, Recall=%.2f%%, Jaccard=%.2f%%, F1 socre=%.2f%%, Initial pts=%d, Final pts=%d\n',...
            mpr*100 ,mre*100, mjacc*100, mf1*100, mpts, mpts_final);
    toc
    infer_save(:,nsource_idx) = [nSource,mpr,mre,mjacc,mf1, mpts,mpts_final];
    %% Save Info
    if save_pred_info
        fclose(label_file);
        fclose(eval_file);
    end
end
