clear
clc

figure()
load('phi_circle_605.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off;
caxis([-1 1])
colorbar('northoutside')
Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);