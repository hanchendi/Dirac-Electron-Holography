clear
clc

subplot('Position',[0.01 0.65 0.23 0.23]);
load('phi_circle_k_60_V_190.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off
caxis([-1 1]);
c=colorbar;
c.Location = 'northoutside';
set(gca,'fontsize',15)
Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[191.0000   24.0000   36.5000]/255,'linewidth',1.5);
text(0.05,1.14,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')

text(0.025,0.9,'(a)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.25 0.65 0.23 0.23]);
load('data_6_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1 1]);
text(0.025,0.9,'(b)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.52 0.65 0.23 0.23]);
load('phi_star_k_85_V_190.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off
caxis([-1.5 1.5]);
c=colorbar;
c.Location = 'northoutside';
set(gca,'fontsize',15)

load('data_star.mat')
r_scale=8;
Nj = 50*4;
Rj=r_scale; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
r=Rj./((1/a*abs(cos(thetj*m/4))).^n2+(1/b*abs(sin(thetj*m/4))).^n3).^(1/n1);
Zj=r.*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[191.0000   24.0000   36.5000]/255,'linewidth',1.5);
text(0.05,1.14,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')

text(0.025,0.9,'(c)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.76 0.65 0.23 0.23]);
load('data_85_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1.5 1.5]);
text(0.025,0.9,'(d)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.17 0.115 0.75 0.5]);

load('data_fidelity.mat')

fill([5.81 6.025 6.025 5.81 5.81],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on
fill([7.04 9.114 9.114 8.04 8.04],[0.8 0.8 1 1 0.8],[207 207 207]/255,'LineStyle','none');hold on

h1=plot(E_choose,F_circle,'linewidth',1.5,'color',[31 119 180]/255);hold on;
h2=plot(E_choose,F_star,'linewidth',1.5,'color',[255 127 14]/255);hold on;
plot(6,0.979,'.','color',[31 119 180]/255,'markersize',20);hold on
plot(8.5,0.9875,'.','color',[255 127 14]/255,'markersize',20);hold on

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'color',[31 119 180]/255);

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((7  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'color',[255 127 14]/255);

axis([5 10 0.8 1])

xlabel('$E$','interpreter','latex')
ylabel('Fidelity','interpreter','latex')
h=legend([h1,h2],'$|\psi_{design}\psi^*_{circle}|$','$|\psi_{design}\psi^*_{star}|$','location','southeast');
set(h,'interpreter','latex')
set(gca,'fontsize',15)

text(-0.2,1,'(e)','Units', 'Normalized','fontsize',17)

set(gcf,'position',[100 100 500 500])