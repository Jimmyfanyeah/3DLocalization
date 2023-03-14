function cPSF=RPSFSpiralPhase(Nzones,Np,L,zeta) 
%
% global Win
% Rotating PSF via linearly increasing spiral-phase winding number in unit step
% from one zone to the next in the aperture ("pupil" and "aperture" are the same)
%% Input variables - 
% Nzones - no. of zones in the circular imaging aperture == number of concentric annular zones in the phase mask (P6)
% Nzones = 7
% zeta - defocus parameter - radians of quadratically varying phase 
         % due to axial defocus, as measured at the edge of pupil
% L = aperture plane side length (in units of aperture radius), > 2
%                              i.e., L/2 = over-sampling factor; 
% L = 4 == the aperture-plane (also known as pupil plane) side length
% Np = array dimension per axis in aperture and image planes (for the FFT)
% Np = 96
%% Output variable
% cPSF - PSF image for centered single source, circshift as needed by actual source location

alpha=1/Nzones;
% side length = L; 
% oversample=side/2; % oversampling factor - critical sampling

% linspace(-2,2,4) = [-2.0000 -0.6667 0.6667 2.0000],
% x = each colum equals to same value from -L/2 to L/2
% y = each row equals to same value from -L/2 to L/2
[x,y]=meshgrid(linspace(-L/2,L/2,Np)); 
[phi,u]=cart2pol(x,y);  % phi: angle, u: magnitude  % [phi, u]=point in pupil plane after scaled
pupilfn=zeros(size(u));
for iu=1:Nzones
%     pupilfn=pupilfn+exp(-1i*(iu)*phi).*(u<sqrt((iu)*alpha) &...
%                      u >= sqrt((iu-1)*alpha));
%  if iu/2 ~= round(iu/2)

    % delta(x_i,y_i)
    pupilfn=pupilfn+exp(-1i*(iu)*(phi+pi/2)).*(u<sqrt((iu)*alpha) &...
                     u >= sqrt((iu-1)*alpha)); % modified version
%  end
end
powereff=1/pi*sum(sum(abs(pupilfn).^2.*(u<1)))*(4/256)^2; % 1 unless
                               % amplitude coding of the pupil as well
% pupilfn=exp(1i*(zeta*u.^2)).*pupilfn.*(u<1);
% PSF=fft2(pupilfn);

pupilfn=exp(-1i*(zeta*u.^2)).*pupilfn.*(u<1); % modified version

% add gaussian noise to PSF
% rng(1);
% sigma = 0.5;
% PSF_error = sigma*randn(size(pupilfn));
% pupilfn = exp(-1i*(zeta*u.^2+PSF_error)).*pupilfn.*(u<1);

%% 0923 neumann bc
% pupilfn_fliplr = fliplr(pupilfn);
% pupilfn_flipud = flipud(pupilfn);
% pupilfn_flip = flipud(pupilfn_fliplr);
% pupilfn_neumann = [pupilfn,pupilfn_fliplr;pupilfn_flipud,pupilfn_flip];
% pupzero = zeros(size(pupilfn));
% pupilfn_neumann = [pupilfn,pupilfn_fliplr;pupilfn_flipud,pupilfn_flip];
% PSF=ifft2(pupilfn_neumann);
% PSF = PSF(1:Np,1:Np);

PSF=ifft2(pupilfn); % modified version
cPSF=abs(fftshift(PSF)).^2; % centered PSF
% cPSF=cPSF/sum(sum(cPSF)); % normalized to unit flux
cPSF=cPSF/norm(cPSF(:)); % normalized to unit flux
% cPSF=cPSF/norm(Cut_fft(cPSF, Win));