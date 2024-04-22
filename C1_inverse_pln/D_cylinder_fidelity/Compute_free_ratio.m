clear
clc

k0=5;
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

f=sum(sum(psiA_target.*conj(psiA_incidence)+psiB_target.*conj(psiB_incidence)))/(64^2);
disp(abs(f))