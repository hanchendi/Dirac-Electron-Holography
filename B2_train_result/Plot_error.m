clear
clc

load('error_train_1.mat')
plot(error_train,'linewidth',1.5);hold on
load('error_val_1.mat')
plot(error_val,'linewidth',1.5);hold on
set(gca,'fontsize',13)
xlabel('Epoch','interpreter','latex')
ylabel('MSE','interpreter','latex')
h=legend('Train','Val');
set(h,'interpreter','latex')
axis([0 1000 0 0.25])