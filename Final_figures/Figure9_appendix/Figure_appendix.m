clear
clc

%%
subplot('Position',[0.1 0.71 0.8 0.285]);

theta=linspace(0,2*pi);
fill(0.05*cos(theta),0.05*sin(theta),[0,0,0]);hold on

x_center=1.5;
y_center=0;
Nj = 200;
Rj=1; 
thetj= linspace(0,2*pi,Nj);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj);
Zj=Zj+x_center+sqrt(-1)*y_center;
fill(real(Zj),imag(Zj),[207 207 207]/255,'linewidth',1.5);hold on

plot([3 5],[-1 -1],'k--');hold on
plot([3 5],[1 1],'k--');hold on
plot([3 3],[-1 1],'k--');hold on
plot([5 5],[-1 1],'k--');hold on
axis([-1 6 -1.6 1.6])
grid on

ylabel('$y$','interpreter','latex')
set(gca,'fontsize',15)

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;

for i=1:8
    x1Annotation = axPos(1) + ((0  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    x2Annotation = axPos(1) + ((0.3*cos(2*pi/8*i)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    y2Annotation = axPos(2) + ((0.3*sin(2*pi/8*i)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);
end

x1Annotation = axPos(1) + ((1.5  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((1.5+1*cos(pi/4)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((1*cos(pi/4)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);

set(gca,'xtick',[-1 0 1 2 3 4 5 6],'xticklabel',[])
text(0.5,0.5,'$r$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(-0.1,0.95,'(a)','Units', 'Normalized','fontsize',17)
%%
subplot('Position',[0.1 0.4 0.8 0.285]);

theta=linspace(0,2*pi);
fill(0.05*cos(theta),0.05*sin(theta),[0,0,0]);hold on

a=0.4;
x_center=1.5;
y_center=0;
Nj = 100*4;
Rj = 0.8; 

thetj1 = linspace(-pi/2, pi/2, Nj/4);
thetj2 = pi/2*ones(1, Nj/4);
thetj3 = linspace(pi/2, 3*pi/2, Nj/4);
thetj4 = -pi/2*ones(1, Nj/4);

zRj1 = a + Rj*exp(sqrt(-1)*thetj1);
zRj2 = linspace(a, -a, length(thetj2)) + Rj*exp(sqrt(-1)*thetj2);
zRj3 = -a + Rj*exp(sqrt(-1)*thetj3);
zRj4 = linspace(-a, a, length(thetj4)) + Rj*exp(sqrt(-1)*thetj4);
Zj = [zRj1 zRj2(2:end) zRj3(2:end) zRj4(2:end-1)];
Zj=Zj+x_center+sqrt(-1)*y_center;

fill(real(Zj),imag(Zj),[207 207 207]/255,'linewidth',1.5);hold on

plot([3 5],[-1 -1],'k--');hold on
plot([3 5],[1 1],'k--');hold on
plot([3 3],[-1 1],'k--');hold on
plot([5 5],[-1 1],'k--');hold on
axis([-1 6 -1.6 1.6])
grid on

ylabel('$y$','interpreter','latex')
set(gca,'fontsize',15)

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;

for i=1:8
    x1Annotation = axPos(1) + ((0  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    x2Annotation = axPos(1) + ((0.3*cos(2*pi/8*i)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    y2Annotation = axPos(2) + ((0.3*sin(2*pi/8*i)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);
end

plot([1.9 1.9],[-0.8 0.8],'k--');hold on
plot([1.1 1.1],[0.8 1],'k');hold on
plot([1.9 1.9],[0.8 1],'k');hold on

x1Annotation = axPos(1) + ((1.5  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((1.9  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);

x1Annotation = axPos(1) + ((1.5  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((1.1  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0.9- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);

x1Annotation = axPos(1) + ((1.9  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + ((1.9+0.8*cos(pi/4)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0.8*sin(pi/4)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);

text(0.5,0.5,'$r$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.5,0.5,'$d$','Units', 'Normalized','fontsize',15,'interpreter','latex')
set(gca,'xtick',[-1 0 1 2 3 4 5 6],'xticklabel',[])
text(-0.1,0.95,'(b)','Units', 'Normalized','fontsize',17)
%%
subplot('Position',[0.1 0.09 0.8 0.285]);

theta=linspace(0,2*pi);
fill(0.05*cos(theta),0.05*sin(theta),[0,0,0]);hold on

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

plot([3 5],[-1 -1],'k--');hold on
plot([3 5],[1 1],'k--');hold on
plot([3 3],[-1 1],'k--');hold on
plot([5 5],[-1 1],'k--');hold on
axis([-1 6 -1.6 1.6])
grid on
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
set(gca,'fontsize',15)

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;

for i=1:8
    x1Annotation = axPos(1) + ((0  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    x2Annotation = axPos(1) + ((0.3*cos(2*pi/8*i)  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
    y1Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    y2Annotation = axPos(2) + ((0.3*sin(2*pi/8*i)- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
    annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation],'HeadLength',5,'HeadWidth',5);
end
text(-0.1,0.95,'(c)','Units', 'Normalized','fontsize',17)

set(gcf,'position',[100 100 500 650])