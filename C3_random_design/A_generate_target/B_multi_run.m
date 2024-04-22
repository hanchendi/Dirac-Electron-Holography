clear
clc

rng(floor(mod(now*10^5,10^5)))
N_single=200;
E_choose=[5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10];
for idx=1:11
    
    V_gather=zeros(N_single,6);
    E_gather=zeros(N_single,1);
    ee_gather=zeros(N_single,1);
    phi_gather=zeros(N_single,4,64,64);
    score_gather=zeros(N_single,1);
    
    for i=1:N_single
        
        V_min=-5;
        V_max=20;
        
        V=rand(1,6)*(V_max-V_min)+V_min;
        E=E_choose(idx);
        [psiA,psiB,score,ee] = MMP_single(E,V);
        
        V_gather(i,:)=V;
        E_gather(i,:)=E;
        ee_gather(i)=ee;
        score_gather(i)=score;
        phi_gather(i,1,:,:)=real(psiA);
        phi_gather(i,2,:,:)=imag(psiA);
        phi_gather(i,3,:,:)=real(psiB);
        phi_gather(i,4,:,:)=imag(psiB);
        disp([idx,i,ee])
    end
    
    save(['data_E_',num2str(idx),'.mat'],'E_gather','V_gather','ee_gather','score_gather','phi_gather')
end
   
