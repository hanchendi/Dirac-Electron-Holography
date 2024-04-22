clear
clc

index_choose=randsample(1:1000,1);
index_choose=4;
load('data_91.mat')
for i=1:1
    subplot(3,1,i)
    temp=phi_gather(index_choose,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
end

load('data_nearest.mat')
for i=1:1
    subplot(3,1,i+1)
    temp=min_dis_phi1(1,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
end

for i=1:1
    subplot(3,1,i+2)
    temp=min_dis_phi2(1,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
end
