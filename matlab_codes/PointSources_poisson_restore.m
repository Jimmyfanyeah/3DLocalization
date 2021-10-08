function Image=PointSources_poisson_restore(nSource,V)
% used in generate img based on predicted 3d loc, 
% not use photon, but flux instead

global Nzones Np L
% nSource = # point sources; V= [Xp, Yp, Tau, Flux] - vectors of x,y,z coords,
%           and flux; Np x Np image array; Nzones: #Fresnel zones

Sources=zeros(Np);

for n=1:nSource
    X=V(n); 
    Y=V(nSource+n);
    Xlower=floor(X); 
    Xupper=Xlower+1; 
    Ylower=floor(Y); 
    Yupper=Ylower+1;
    %     PSF=RotatingPSF(Nzones,V(2*nSource+n),Np);
    PSF = RPSFSpiralPhase(Nzones,Np,L,V(2*nSource+n));
    
    % do bilinear interpolation on the PSF
    shift=[Xlower Ylower];
    PSF11=circshift(PSF,shift);
    PSF21=circshift(PSF11,[1 0]);
    PSF12=circshift(PSF11,[0 1]);
    PSF22=circshift(PSF12,[1 0]);
    IPSF=(Xupper-X)*(PSF11*(Yupper-Y)+PSF12*(Y-Ylower))+...
        (X-Xlower)*(PSF21*(Yupper-Y)+PSF22*(Y-Ylower)); % bilinearly interpolated PSF
    Sources=Sources+V(3*nSource+n)*IPSF;
%     disp(['IPSF sum:',num2str(sum(IPSF(:))),' max:',num2str(max(IPSF(:))),' max2:',num2str(max(IPSF(:))/sum(IPSF(:)))])
end
Image=Sources;
% figure(10);imshow(Image,[])
end