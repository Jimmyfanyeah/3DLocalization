function [center, dx, dy, dz] = local_3Dmax_large(u1)
% clustered non-zero elements are considered as identified particles 
% and 3D averaged positions are calculated
% center, dx, dy, dz are all 3D matrix with the same size as u1
% center records center position of each particle
% dx, dy, dz records position corrections in x, y, and z dimensions.
[a,b,c] = size(u1);

[I, q] = max(u1(:)); % I = max value q = index
[X, Y, Z] = ndgrid(1:a, 1:b, 1:c);
center = zeros(a,b,c);
dx = center;
dy = center;
dz = center;
while I>0
    [xc, yc, zc] = ind2sub([a,b,c], q);
    inx3 = zeros(a,b,c);
    inx3(q) = 1;
    % compute the centroid for one point source
    inx3 = label_neibs(u1, inx3, xc, yc, zc, a, b, c);
    inx3 = inx3>0;
    photons = sum(u1(inx3));
    elx = sum(X(inx3).*u1(inx3))/photons;
    ely = sum(Y(inx3).*u1(inx3))/photons;
    elz = sum(Z(inx3).*u1(inx3))/photons;

    center(round(elx),round(ely),round(elz)) = photons;
    dx(round(elx),round(ely),round(elz)) = elx-round(elx); % difference of centroid 
    dy(round(elx),round(ely),round(elz)) = ely-round(ely);
    dz(round(elx),round(ely),round(elz)) = elz-round(elz);
    u1(inx3) = 0;
    [I, q] = max(u1(:));
end
% remove some dim point
thread = max(center(:))*0.05; 
dx(center<=thread) = 0;
dy(center<=thread) = 0;
dz(center<=thread) = 0;
center(center<=thread) = 0;
end

function inx3 = label_neibs(h2d, inx3, xc, yc, zc, a, b, c)
% find the neighbohood of (yc, xc, zc) such that (yc, xc, zc) is 
% brightest point and other points (inx3 is not zero) is above zero
% output is index of all marked points
[dx, dy, dz] = ndgrid(xc-2:xc+2,yc-2:yc+2, zc-1:zc+1);
dx(38) = []; dy(38) = []; dz(38) = []; % the position of inx3
inx = (dx>0&dx<=a)&(dy>0&dy<=b)&(dz>0&dz<=c);
dx = dx(inx); dy = dy(inx); dz = dz(inx);

for k = 1:numel(dy)
    if(inx3(dx(k),dy(k),dz(k))==0 && h2d(dx(k),dy(k),dz(k))>0 && h2d(dx(k),dy(k),dz(k))<=h2d(xc,yc,zc))
        inx3(dx(k),dy(k),dz(k)) = 1;
        inx3 = label_neibs(h2d, inx3, dx(k),dy(k),dz(k),a,b,c);
    end
end
end
