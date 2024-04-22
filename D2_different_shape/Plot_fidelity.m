clear
clc

load('data_fidelity.mat')

plot(E_choose,F_circle,'linewidth',1.5);hold on;
plot(E_choose,F_star,'linewidth',1.5);hold on

plot([6,6],[0,1],'r--');hold on
plot([9,9],[0,1],'r--');hold on
axis([5 10 0.8 1])

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
h=legend('$|\psi_{design}\psi^*_{circle}|$','$|\psi_{design}\psi^*_{star}|$');
set(h,'interpreter','latex')
set(gca,'fontsize',15)