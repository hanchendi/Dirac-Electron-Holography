clear
clc

rng(floor(mod(now*10^5,10^5)))
N_single=1;
for idx=1:1
    
    V_gather=zeros(N_single,6);
    E_gather=zeros(N_single,1);
    ee_gather=zeros(N_single,1);
    phi_gather=zeros(N_single,4,64,64);
    score_gather=zeros(N_single,1);
    
    for i=1:N_single

        %% Generate energy range from 5 to 10 with voltage from -5 to 20.
        E_min=5;
        E_max=10;
        V_min=-5;
        V_max=20;
        
        V=rand(1,6)*(V_max-V_min)+V_min;
        E=rand()*(E_max-E_min)+E_min;
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
    
    save(['data_',num2str(idx),'.mat'],'E_gather','V_gather','ee_gather','score_gather','phi_gather')
end
   
