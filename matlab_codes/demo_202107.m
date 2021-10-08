% clear; close all
global   Np L Nzones nSource
L = 4; % pupil radius = #R aperture plane side length (in units of aperture radius), > 2
Nzones = 7; % no. of zones in the circular imaging aperture = #L
% nSource = 30; 
Np = 96;
% Ntest  = [0,10000];
Ntest  = [1,50];

% imshow(g,[])
% imagesc(g)
% colorbar

save_dir = '/home/tonielook/Documents/3dloc/test_v2/test25/';
if ~exist(save_dir, 'dir')
   mkdir(save_dir)
end

% nSource is random values uniformly from [5,40]
% all_nSource = randi([5,40],[Ntest(2),1]);
% all_nSource = randi([5,40],[Ntest(2),1]);

%% ground true and observed image not on grid point
all_flux = [];
all_nSource_v1 = [];
label_file = fopen([save_dir,'label.txt'],'w');
for ii = Ntest(1):Ntest(2)
    rng(ii);
%     nSource = all_nSource(ii+1);
    nSource = 25;
    Xp_true = 34*2*(rand(1,nSource)-0.5);
    Yp_true = 34*2*(rand(1,nSource)-0.5);
    zeta_true = 2*20*(rand(1,nSource)-0.5);   % in original file is 20
 
    % check overlap
    kk=1;
    while kk <= length(Xp_true)
        qq = kk + 1;
        while qq <= length(Xp_true)
            if abs(Xp_true(kk)-Xp_true(qq))<=2 && abs(Yp_true(kk)-Yp_true(qq))<=2 && abs(zeta_true(kk)-zeta_true(qq))<=1
                Xp_true(qq) = [];
                Yp_true(qq) = [];
                zeta_true(qq) = [];
                disp([num2str(ii), ' point ',num2str(kk) ' and ' num2str(qq)])
            else
                qq = qq+1;
            end
        end
        kk = kk+1;
    end

    Flux_true = poissrnd(2000,[1,length(Xp_true)]); % Random numbers from Poisson distribution 

    Vtrue = [Xp_true Yp_true zeta_true Flux_true];
    [I0,flux] = PointSources_poisson_v2(length(Xp_true),Vtrue); % flux value in normalized basis case
%     figure; imshow(I0,[])

    b = 5;
    g = poissrnd(I0+b); % Obversed image with poisson noise
%     figure;imshow(g,[])

    all_flux = [all_flux; flux];

    % use gloabl factor to normalize
    scale_factor = [b,130];
    scaled_g = (g - scale_factor(1))/scale_factor(2);

%     imwrite(scaled_g,[save_dir,'\im',num2str(ii),'.png'])
    
    % save mat file
    save([save_dir,'im',num2str(ii),'.mat'],'g');
    disp([num2str(ii),' saved']);
    % save labels
    LABEL = [ii*ones(1,length(Xp_true)); Yp_true; Xp_true; zeta_true; flux'];
    fprintf(label_file,'%d %6.4f %6.4f %6.4f %6.4f \n',LABEL);

end
fclose(label_file);

% visualize 3d plot
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

