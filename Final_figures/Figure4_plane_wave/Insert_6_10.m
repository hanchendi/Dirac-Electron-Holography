clear
clc

%%
k0=6;
s0=1;
N=64;
x_choose=linspace(3,5,N);
y_choose=linspace(-1,1,N);
[xx,yy]=meshgrid(x_choose,y_choose);
zz=xx+sqrt(-1)*yy;

psiA_target=exp(sqrt(-1)*k0*xx)/sqrt(2);
psiB_target=s0*exp(sqrt(-1)*k0*xx)/sqrt(2);
norm_target=sum(sum(abs(psiA_target).^2+abs(psiB_target).^2))/N^2;
psiA_target=psiA_target./sqrt(norm_target);
psiB_target=psiB_target./sqrt(norm_target);

psiA_incidence=besselh(0,1,abs(zz)*k0)/sqrt(2);
psiB_incidence=s0*sqrt(-1)*besselh(1,1,abs(zz)*k0).*exp(sqrt(-1)*angle(zz))/sqrt(2);
norm_incidence=sum(sum(abs(psiA_incidence).^2+abs(psiB_incidence).^2))/N^2;
psiA_incidence=psiA_incidence./sqrt(norm_incidence);
psiB_incidence=psiB_incidence./sqrt(norm_incidence);

figure()
subplot('Position',[0 0 1 1]);
temp=real(psiA_incidence);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis([1 64 1 64])
set(gcf,'position',[100 100 500 500])

%%
k0=10;
s0=1;
N=64;
x_choose=linspace(3,5,N);
y_choose=linspace(-1,1,N);
[xx,yy]=meshgrid(x_choose,y_choose);
zz=xx+sqrt(-1)*yy;

psiA_target=exp(sqrt(-1)*k0*xx)/sqrt(2);
psiB_target=s0*exp(sqrt(-1)*k0*xx)/sqrt(2);
norm_target=sum(sum(abs(psiA_target).^2+abs(psiB_target).^2))/N^2;
psiA_target=psiA_target./sqrt(norm_target);
psiB_target=psiB_target./sqrt(norm_target);

psiA_incidence=besselh(0,1,abs(zz)*k0)/sqrt(2);
psiB_incidence=s0*sqrt(-1)*besselh(1,1,abs(zz)*k0).*exp(sqrt(-1)*angle(zz))/sqrt(2);
norm_incidence=sum(sum(abs(psiA_incidence).^2+abs(psiB_incidence).^2))/N^2;
psiA_incidence=psiA_incidence./sqrt(norm_incidence);
psiB_incidence=psiB_incidence./sqrt(norm_incidence);

figure()
subplot('Position',[0 0 1 1]);
temp=real(psiA_incidence);
temp=reshape(temp,[64,64]);
pcolor(temp)
shading flat;
axis off
set(gcf,'position',[100 100 500 500])