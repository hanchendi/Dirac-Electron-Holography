clear
clc

%%
load('phi_circle_8.mat')
subplot('Position',[0.05 0.615 0.35 0.29]);
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off
colorbar('northoutside')
set(gca,'fontsize',15)
caxis([-1.5 1.5])

Nj = 200;
Rj=7; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
Zj=Rj*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);
text(0,0.95,'(a1)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.45 0.615 0.35 0.29]);
load('data_8_circle_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
set(gca,'fontsize',15)
caxis([-1.5 1.5])
text(0,0.95,'(a2)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.63 0.938 0.35 0.02]);

red_yellow_blue=[166 0 38;
    216 48 35;
    246 110 68;
    250 172 93;
    255 223 147;
    255 255 189;
    222 244 249;
    171 217 233;
    115 173 210;
    72 115 181;
    49 54 145];

blue_yellow_red=flip(red_yellow_blue,1);
index=0;
for i=1:10
    for j=0:99
        color255=[0 0 0];
        for k=1:3
            color255(k)=color255(k)+j/100*blue_yellow_red(i+1,k)+(100-j)/100*blue_yellow_red(i,k);
        end
        for k=1:3
            color255(k)=min(color255(k),255);
        end
        fill([index index+1 index+1 index],[0 0 1 1],color255/255,'LineStyle','none');hold on
        index=index+1;
    end
end
axis([0 1000 0 1])
yticks([])
xticks([0 1000])
xticklabels({'-5','20'})
set(gca,'fontsize',14)
set(gca,'xaxisLocation','top')

axPos = get(gca,'Position');
xMinMax = xlim;
yMinMax = ylim;
x1Annotation = axPos(1) + (((8+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + (((8+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((-0.01- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

subplot('Position',[0.8 0.615 0.2 0.25]);

V_min=-5;
V_max=20;
theta=linspace(0,2*pi);
index=1;
for j=1:3
    for i=1:2
        v_norm=(V(index)-V_min)/(V_max-V_min);
        for k=1:10
            if k/10>v_norm
                r1=(v_norm-(k-1)/10)*10;
                r2=(k/10-v_norm)*10;
                color=blue_yellow_red(k,:)*(1-r1)+blue_yellow_red(k+1,:)*(1-r2);
                break
            end
        end
        for k=1:3
            color(k)=min(color(k),255);
        end
        fill(0.35*cos(theta)+i,0.35*sin(theta)+j-2,color/255);hold on
        index=index+1;
    end
end
axis([0.5 2.5 -1.45 1.55])
axis off
text(0.05,1.1,'(a3)','Units', 'Normalized','fontsize',17)

%%
load('phi_stadium_8.mat')
subplot('Position',[0.05 0.31 0.35 0.29]);
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off
set(gca,'fontsize',15)
caxis([-1.5 1.5])

Nj = 50*4;
Rj = 0.8; 
a=0.4;
thetj1 = linspace(-pi/2, pi/2, Nj/4);
thetj2 = pi/2*ones(1, Nj/4);
thetj3 = linspace(pi/2, 3*pi/2, Nj/4);
thetj4 = -pi/2*ones(1, Nj/4);

zRj1 = a + Rj*exp(sqrt(-1)*thetj1);
zRj2 = linspace(a, -a, length(thetj2)) + Rj*exp(sqrt(-1)*thetj2);
zRj3 = -a + Rj*exp(sqrt(-1)*thetj3);
zRj4 = linspace(-a, a, length(thetj4)) + Rj*exp(sqrt(-1)*thetj4);
Zj = [zRj1 zRj2(2:end) zRj3(2:end) zRj4(2:end-1)]*8+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);
text(0,0.95,'(b1)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.45 0.31 0.35 0.29]);
load('data_8_stadium_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
set(gca,'fontsize',15)
caxis([-1.5 1.5])
text(0,0.95,'(b2)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.8 0.31 0.2 0.25]);

V_min=-5;
V_max=20;
theta=linspace(0,2*pi);
index=1;
for j=1:3
    for i=1:2
        v_norm=(V(index)-V_min)/(V_max-V_min);
        for k=1:10
            if k/10>v_norm
                r1=(v_norm-(k-1)/10)*10;
                r2=(k/10-v_norm)*10;
                color=blue_yellow_red(k,:)*(1-r1)+blue_yellow_red(k+1,:)*(1-r2);
                break
            end
        end
        for k=1:3
            color(k)=min(color(k),255);
        end
        fill(0.35*cos(theta)+i,0.35*sin(theta)+j-2,color/255);hold on
        index=index+1;
    end
end
axis([0.5 2.5 -1.45 1.55])
axis off
text(0.05,1.1,'(b3)','Units', 'Normalized','fontsize',17)

%%
load('phi_star_8.mat')
subplot('Position',[0.05 0.005 0.35 0.29]);
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);hold on
shading flat;
axis off
set(gca,'fontsize',15)
caxis([-1.5 1.5])

load('data_star.mat')
r_scale=8;
Nj = 50*4;
Rj=r_scale; 
thetj= linspace(0,2*pi,Nj+1);
thetj = thetj(1:Nj);
r=Rj./((1/a*abs(cos(thetj*m/4))).^n2+(1/b*abs(sin(thetj*m/4))).^n3).^(1/n1);
Zj=r.*exp(sqrt(-1)*thetj)+53+55*sqrt(-1);
fill(real(Zj),imag(Zj),[166 0 38]/255,'linewidth',1.5);

text(0,0.95,'(c1)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.45 0.005 0.35 0.29]);
load('data_8_star_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp);
shading flat;
axis off
set(gca,'fontsize',15)
caxis([-1.5 1.5])

text(0,0.95,'(c2)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.8 0.005 0.2 0.25]);

V_min=-5;
V_max=20;
theta=linspace(0,2*pi);
index=1;
for j=1:3
    for i=1:2
        v_norm=(V(index)-V_min)/(V_max-V_min);
        for k=1:10
            if k/10>v_norm
                r1=(v_norm-(k-1)/10)*10;
                r2=(k/10-v_norm)*10;
                color=blue_yellow_red(k,:)*(1-r1)+blue_yellow_red(k+1,:)*(1-r2);
                break
            end
        end
        for k=1:3
            color(k)=min(color(k),255);
        end
        fill(0.35*cos(theta)+i,0.35*sin(theta)+j-2,color/255);hold on
        index=index+1;
    end
end
axis([0.5 2.5 -1.45 1.55])
axis off
text(0.05,1.1,'(c3)','Units', 'Normalized','fontsize',17)

text(0.05,1.14,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.05,1.14,'$V$','Units', 'Normalized','fontsize',15,'interpreter','latex')
set(gcf,'position',[100 100 500 600])