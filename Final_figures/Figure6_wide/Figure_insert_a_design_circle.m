clear
clc

figure()
load('data_wide_75_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);
caxis([-1.3 1.3])
shading flat;
axis off

