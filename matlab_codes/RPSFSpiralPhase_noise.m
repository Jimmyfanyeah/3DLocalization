function cPSF=RPSFSpiralPhase_noise(Nzones,Np,L,zeta,noiselevel) 
%
% global Win
% Rotating PSF via linearly increasing spiral-phase winding number in unit step
% from one zone to the next in the aperture ("pupil" and "aperture" are the same)
%% Input variables - 
% Nzones - no. of zones in the circular imaging aperture
% zeta - defocus parameter - radians of quadratically varying phase 
         % due to axial defocus, as measured at the edge of pupil
% L = aperture plane side length (in units of aperture radius), > 2
%                              i.e., L/2 = over-sampling factor; 
% Np = array dimension per axis in aperture and image planes (for the FFT)
%% Output variable
% cPSF - PSF image for centered single source, circshift as needed by actual source location
%
alpha=1/Nzones;
% side length = L; 
% oversample=side/2; % oversampling factor - critical sampling 
[x,y]=meshgrid(linspace(-L/2,L/2,Np)); 
[phi,u]=cart2pol(x,y);
pupilfn=zeros(size(u));
rng(2018);
for iu=1:Nzones
%     pupilfn=pupilfn+exp(-1i*(iu)*phi).*(u<sqrt((iu)*alpha) &...
%                      u >= sqrt((iu-1)*alpha));
%  if iu/2 ~= round(iu/2)
    pupilfn=pupilfn+exp(-1i*((iu)*(phi+pi/2)+2*pi*noiselevel*randn(size(phi)))).*(u<sqrt((iu)*alpha) &...
                     u >= sqrt((iu-1)*alpha)); % modified version
%  end
end
powereff=1/pi*sum(sum(abs(pupilfn).^2.*(u<1)))*(4/256)^2; % 1 unless
                               % amplitude coding of the pupil as well
% pupilfn=exp(1i*(zeta*u.^2)).*pupilfn.*(u<1);
% PSF=fft2(pupilfn);

pupilfn=exp(-1i*(zeta*u.^2)).*pupilfn.*(u<1); % modified version
% pupilfn=exp(-1i*(zeta*u.^2)).*pupilfn;
% pupilfn = pupilfn + noiselevel*max(pupilfn(:))*randn(size(pupilfn)); 
% pupilfn = pupilfn.*(u<1);
PSF=ifft2(pupilfn); % modified version
cPSF=abs(fftshift(PSF)).^2; % centered PSF
% cPSF=cPSF/sum(sum(cPSF)); % normalized to unit flux
cPSF=cPSF/norm(cPSF(:)); % normalized to unit flux
% cPSF=cPSF/norm(Cut_fft(cPSF, Win));