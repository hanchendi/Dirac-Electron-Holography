clear
clc

figure()
load('data_wide_7_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1 1])