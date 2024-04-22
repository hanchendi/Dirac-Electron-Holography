clear
clc

%%
subplot('Position',[0.07 0.12 0.9 0.85]);
theta=linspace(0,2*pi);
fill(0.05*cos(theta),0.05*sin(theta),[0,0,0]);hold on

plot([3 5],[-1 -1],'k--');hold on
plot([3 5],[1 1],'k--');hold on
plot([3 3],[-1 1],'k--');hold on
plot([5 5],[-1 1],'k--');hold on
axis([-0.5 5.5 -1.6 1.6])
plot([4-pi/8 4+pi/8],[1.25 1.25],'k','linewidth',2);hold on
axis off

load('data_star.mat')
r_scale=1.2;
x_center=1.5;
y_center=0;
Nj = 100*4;
Rj=r_scale; 

thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
r=Rj./((1/a*abs(cos(thetj*m/4))).^n2+(1/b*abs(sin(thetj*m/4))).^n3).^(1/n1);
Zj=r.*exp(sqrt(-1)*thetj);
Zj=Zj+x_center+sqrt(-1)*y_center;
fill(real(Zj),imag(Zj),[207 207 207]/255,'linewidth',1.5);hold on

set(gca,'fontsize',15)

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;

for i=1:8
    x1Annotation = axPos(1) + ((0  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    x2Annotation = axPos(1) + ((0.3*cos(2*pi/8*i)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    y2Annotation = axPos(2) + ((0.3*sin(2*pi/8*i)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',7,'HeadWidth',7);
end

x1Annotation = axPos(1) + ((-0.25  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((0.5  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((-1.25- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((-1.25- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',7,'HeadWidth',7);

x1Annotation = axPos(1) + ((-0.25  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((-0.25  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((-1.25- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((-0.5- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',7,'HeadWidth',7);

text(0.5,0.5,'$x$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.5,0.5,'$y$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.5,0.5,'$\lambda$','Units', 'Normalized','fontsize',15,'interpreter','latex')
set(gcf,'position',[100 100 500 280])

