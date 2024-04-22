clear
clc

subplot('Position',[0.25 0.84 0.23 0.1437]);

load('phi_circle_75.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
caxis([-1.3 1.3])
shading flat;
axis off;
Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);
text(0,0.9,'(a)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.49 0.84 0.23 0.1437]);

load('data_wide_75_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);
caxis([-1.3 1.3])
shading flat;
axis off
text(0,0.9,'(b)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.15 0.58 0.8 0.23]);
load('data_fidelity_a.mat')

fill([7.369 7.536 7.536 7.369 7.369],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
plot(E_choose,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
plot(7.5,0.98,'k.','markersize',20)
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
colorbar('eastoutside')
caxis([-1.3 1.3])
text(-0.17,1,'(c)','Units', 'Normalized','fontsize',17)
%%

subplot('Position',[0.01 0.35 0.23 0.1437]);

load('phi_circle_605.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off;
caxis([-1 1])
Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);
text(0,0.9,'(d)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.25 0.35 0.23 0.1437]);

load('data_wide_605_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1 1])
text(0,0.9,'(e)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.52 0.35 0.23 0.1437]);

load('phi_circle_7.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off;
caxis([-1 1])
Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);
text(0,0.9,'(f)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.76 0.35 0.23 0.1437]);

load('data_wide_7_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1 1])
text(0,0.9,'(g)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.15 0.09 0.8 0.23]);
load('data_fidelity_b.mat')

fill([6 6.243 6.243 6 6],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
fill([6.32 7.05 7.05 6.32 6.32],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
plot(E_choose-0.05,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
plot(7,0.985,'k.','markersize',20);hold on
plot(6,0.975,'k.','markersize',20)
axis([5 10 0.8 1])

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
set(gca,'fontsize',15)
text(-0.17,1,'(h)','Units', 'Normalized','fontsize',17)
caxis([-1 1])
colorbar('eastoutside')

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
set(gcf,'position',[100 100 500 800])