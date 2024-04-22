clear
clc

subplot('Position',[0.175 0.58 0.775 0.4]);

load('error_train_1.mat')
plot(error_train,'linewidth',1.5);hold on
load('error_val_1.mat')
plot(error_val,'linewidth',1.5);hold on
set(gca,'fontsize',13)
xlabel('Epoch','interpreter','latex')
ylabel('MSE','interpreter','latex')
h=legend('Train','Validate');
set(h,'interpreter','latex')
axis([0 1000 0 0.25])
set(gca,'fontsize',15)
text(-0.21,1,'(a)','Units', 'Normalized','fontsize',17)

%%
index_choose=4;
load('y_test_pred_1.mat')
load('data_91.mat')

subplot('Position',[0.025 0.28 0.23 0.192]);
temp=phi_gather(index_choose,1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(b1)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.265 0.28 0.23 0.192]);
temp=phi_gather(index_choose,2,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(b2)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.505 0.28 0.23 0.192]);
temp=phi_gather(index_choose,3,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(b3)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.745 0.28 0.23 0.192]);
temp=phi_gather(index_choose,4,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(b4)','Units', 'Normalized','fontsize',17)

%%
subplot('Position',[0.025 0.05 0.23 0.192]);
temp=y_test_pred(index_choose,1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(c1)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.265 0.05 0.23 0.192]);
temp=y_test_pred(index_choose,2,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(c2)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.505 0.05 0.23 0.192]);
temp=y_test_pred(index_choose,3,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(c3)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.745 0.05 0.23 0.192]);
temp=y_test_pred(index_choose,4,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
text(0.025,0.9,'(c4)','Units', 'Normalized','fontsize',17)

set(gcf,'position',[100 100 500 600])