clear
clc

subplot('Position',[0.15 0.58 0.8 0.39]);
load('data_fidelity_a.mat')

fill([7.45 7.55 7.55 7.45 7.45],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
plot(E_choose,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
plot(7.5,0.98,'k*','markersize',10)
axis([5 10 0.8 1])
text(0.5,0.5,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')
axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
set(gca,'fontsize',15)
colorbar('northoutside')
caxis([-1.3 1.3])
text(-0.17,1,'(a)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.15 0.09 0.8 0.39]);
load('data_fidelity_b.mat')

fill([6 7 7 6 6],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
plot(E_choose-0.05,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
plot(7,0.985,'k*','markersize',10);hold on
plot(6,0.975,'k*','markersize',10)
axis([5 10 0.8 1])

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
set(gca,'fontsize',15)
text(-0.17,1,'(b)','Units', 'Normalized','fontsize',17)
caxis([-1 1])
colorbar('northoutside')

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

text(0.5,0.5,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')
set(gcf,'position',[100 100 500 600])