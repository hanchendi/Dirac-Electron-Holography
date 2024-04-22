clear
clc

load('data_fidelity.mat')

fill([6 7 7 6 6],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
plot(E_choose,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
axis([5 10 0.8 1])

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
set(gca,'fontsize',15)