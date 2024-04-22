clear
clc

rng(floor(mod(now*10^5,10^5)))
    
phi=zeros(4,64,64);
score_gather=0;

E=10;
V=[ 0.0826141   0.49327295  7.38519428  1.93001801 -0.47027141 12.48324335];
[psiA,psiB,score,ee] = MMP_single(E,V);

phi(1,:,:)=real(psiA);
phi(2,:,:)=imag(psiA);
phi(3,:,:)=real(psiB);
phi(4,:,:)=imag(psiB);
    
save(['data_10_design.mat'],'E','V','ee','phi')

   
