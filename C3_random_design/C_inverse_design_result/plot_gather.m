clear
clc

%% Energy range from 5 to 10
E_choose=[5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10];
F_final_avr=zeros(1,11);
F_final_val=zeros(1,11);
for i=1:11
    load(['data_final_',num2str(i),'.mat'])
    F_final_avr(i)=mean(F_final);
    F_final_val(i)=var(F_final);
end

errorbar(E_choose,F_final_avr,F_final_val,'linewidth',1.5,'color',[31 119 180]/255);hold on
plot(E_choose,F_final_avr,'.','markersize',20,'color',[31 119 180]/255);hold on
