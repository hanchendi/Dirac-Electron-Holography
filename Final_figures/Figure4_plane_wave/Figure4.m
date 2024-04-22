clear
clc

load('data_6.mat')
subplot('Position',[0.05 0.51 0.35 0.35]);
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1/sqrt(2)-0.05 1/sqrt(2)+0.05])
colorbar('northoutside')
set(gca,'fontsize',15)
text(0,0.95,'(a)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.45 0.51 0.35 0.35]);
load('data_6_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1/sqrt(2)-0.05 1/sqrt(2)+0.05])
set(gca,'fontsize',15)
text(0,0.95,'(b)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.63 0.9 0.35 0.0245]);

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
x1Annotation = axPos(1) + (((6+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + (((6+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((-0.01- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

subplot('Position',[0.8 0.5 0.2 0.3]);

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
        fill(0.35*cos(theta)+i,0.35*sin(theta)+j-2,color/255);hold on
        index=index+1;
    end
end
axis([0.5 2.5 -1.45 1.55])
axis off
text(0.05,1.14,'(c)','Units', 'Normalized','fontsize',17)

load('data_10.mat')
subplot('Position',[0.05 0.01 0.35 0.35]);
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1/sqrt(2)-0.05 1/sqrt(2)+0.05])
colorbar('northoutside')
set(gca,'fontsize',15)
text(0,0.95,'(d)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.45 0.01 0.35 0.35]);
load('data_10_design.mat')
temp=phi(1,:,:);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
caxis([-1/sqrt(2)-0.05 1/sqrt(2)+0.05])
set(gca,'fontsize',15)
text(0,0.95,'(e)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.63 0.4 0.35 0.0245]);

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
x1Annotation = axPos(1) + (((10+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
x2Annotation = axPos(1) + (((10+5)/25*1000  - xMinMax(1))/(xMinMax(2)-xMinMax(1))) * axPos(3);
y1Annotation = axPos(2) + ((-0.01- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
y2Annotation = axPos(2) + ((0- yMinMax(1))/(yMinMax(2)-yMinMax(1))) * axPos(4);
annotation('textarrow',[x1Annotation x2Annotation],[y1Annotation y2Annotation]);

subplot('Position',[0.8 0 0.2 0.3]);

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
        fill(0.35*cos(theta)+i,0.35*sin(theta)+j-2,color/255);hold on
        index=index+1;
    end
end
axis([0.5 2.5 -1.45 1.55])
axis off
text(0.05,1.14,'(f)','Units', 'Normalized','fontsize',17)

text(0.05,1.14,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.05,1.14,'Re $\psi_1$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.05,1.14,'$V$','Units', 'Normalized','fontsize',15,'interpreter','latex')
text(0.05,1.14,'$V$','Units', 'Normalized','fontsize',15,'interpreter','latex')
set(gcf,'position',[100 100 500 500])