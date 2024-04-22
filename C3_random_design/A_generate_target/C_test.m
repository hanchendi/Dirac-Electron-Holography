clear
clc

N=64;
x_choose=linspace(4,6,N);
y_choose=linspace(-1,1,N);
[xx,yy]=meshgrid(x_choose,y_choose);

load('data_1.mat')
i_test=1;
figure()
for i=1:4
    subplot(2,2,i)
    temp= phi_gather(i_test,i,:,:);
    temp=reshape(temp,[64,64]);
    pcolor(xx, yy, temp); hold on;
    shading flat;
    colorbar
end

disp(V_gather(i_test,:))
disp(ee_gather(i_test))
disp(score_gather(i_test))
