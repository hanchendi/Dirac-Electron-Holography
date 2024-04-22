clear
clc

subplot('Position',[0.2 0.38 0.6 0.6]);
Ni=40;
No=44;
Nj=(Ni+No)*3;

delta_in=0.1;
delta_out=0.1;

Zi=[];
Zo=[];
Zj=[];

theta_i=linspace(0,2*pi,Ni+1);
theta_o=linspace(0,2*pi,No+1);
theta_j=linspace(0,2*pi,Nj+1);

theta_i=theta_i(1:Ni);
theta_o=theta_o(1:No);
theta_j=theta_j(1:Nj);
rj=0.35;
ri=(1-delta_in)*rj;
ro=(1+delta_out)*rj;

for j=1:3
    for i=1:2
        
        xc=i;
        yc=j-2;
        
        x=xc+ri*cos(theta_i);
        y=yc+ri*sin(theta_i);
        Zi=[Zi x+sqrt(-1)*y];
        
        x=xc+ro*cos(theta_o);
        y=yc+ro*sin(theta_o);
        Zo=[Zo x+sqrt(-1)*y];
        
        x=xc+rj*cos(theta_j);
        y=yc+rj*sin(theta_j);
        Zj=[Zj x+sqrt(-1)*y];
        plot(x,y,'k');hold on
    end
end

plot(real(Zi), imag(Zi),'bo');hold on
plot(real(Zo),imag(Zo),'r*');hold on
axis([0.5 2.5 -1.5 1.5])
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
grid on
set(gca,'fontsize',15)
text(-0.325,1,'(a)','Units', 'Normalized','fontsize',17)

subplot('Position',[0.1 0.09 0.8 0.2]);
load('ee_plot.mat')
histogram(ee_plot,20);
xlabel('MMP error','interpreter','latex')
ylabel('Count','interpreter','latex')
set(gca,'fontsize',15)

text(-0.12,1,'(b)','Units', 'Normalized','fontsize',17)
set(gcf,'position',[100 100 500 750])