clear
clc

index_choose=randsample(1:1000,1);
index_choose=4;
% for index_choose=1:1000

load('data_91.mat')
for i=1:4
    subplot(2,4,i)
    temp=phi_gather(index_choose,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
end

load('y_test_pred_1.mat')
for i=1:4
    subplot(2,4,i+4)
    temp=y_test_pred(index_choose,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(temp)
    shading flat;colorbar
end

mse_error=mean((phi_gather(index_choose,:,:,:)-y_test_pred(index_choose,:,:,:)).^2,[1,2,3,4]);
disp(index_choose)
disp(mse_error)
%end