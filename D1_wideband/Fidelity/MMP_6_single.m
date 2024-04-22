function [psiA,psiB,score,ee] = MMP_6_single(E,V)
    
    load('data_poles.mat')
    valley_index=1;

    s=zeros(1,6);
    k=zeros(1,6);
    k0=abs(E);
    s0=sign(E);

    for i=1:6
        s(i)=sign(E-V(i));
        k(i)=abs(E-V(i));
    end

    pln=[];
    for i=1:6

        pln=[pln besselh(0,1,abs(Zj((i-1)*Nj+1:i*Nj))*k0)/sqrt(2)];
        pln=[pln sqrt(-1)*besselh(1,1,abs(Zj((i-1)*Nj+1:i*Nj))*k0).*exp(sqrt(-1)*angle(Zj((i-1)*Nj+1:i*Nj)))/sqrt(2)];

    end
    pln=conj([pln]');

    L=1;
    nL=-L:L;
    M=[];
    for i=1:6 % boundary 1-6

        M1_temp=[];
        M2_temp=[];
        % inside poles
        for j=1:6

            eval(['RR=R_',num2str(j),'_',num2str(i),';']);
            eval(['PP=Phi_',num2str(j),'_',num2str(i),';']);

            for l=1:length(nL)

                H1 = besselh(nL(l), 1, k0*RR).*exp(sqrt(-1)*nL(l)*PP);
                H2 = sqrt(-1)^valley_index*s0*besselh(nL(l)+valley_index, 1, k0*RR).*exp(sqrt(-1)*(nL(l)+valley_index)*PP);
                M1_temp = [M1_temp -H1];
                M2_temp = [M2_temp -H2];

            end
        end

        % outside poles
        for j=1:6

            if i==j
                eval(['RR=R_',num2str(i),';']);
                eval(['PP=Phi_',num2str(i),';']);
                for l=1:length(nL)
                    H1 = besselh(nL(l), 1, k(j)*RR).*exp(sqrt(-1)*nL(l)*PP);
                    H2 = sqrt(-1)^valley_index*s(j)*besselh(nL(l)+valley_index, 1, k(j)*RR).*exp(sqrt(-1)*(nL(l)+valley_index)*PP);
                    M1_temp = [M1_temp H1];
                    M2_temp = [M2_temp H2];
                end
            else
                M1_temp = [M1_temp zeros(Nj,length(nL)*No)];
                M2_temp = [M2_temp zeros(Nj,length(nL)*No)];
            end
        end

        M=[M;M1_temp;M2_temp];

    end

    C = pinv(M)*pln;
    ee = norm(M*C - pln)/norm(pln);

    N=64;
    x_choose=linspace(3,5,N);
    y_choose=linspace(-1,1,N);

    [xx,yy]=meshgrid(x_choose,y_choose);
    zz=xx+sqrt(-1)*yy;

    psiA=zeros(N,N);
    psiB=zeros(N,N);

    % inside poles, outside wave
    t=1;
    for idx=1:6
        for i=1:length(nL)

            index_p=(idx-1)*Ni+1;
            for j=1:Ni

                Z_p=(zz-Zi(index_p));
                r_p=abs(Z_p);
                theta_p=angle(Z_p);

                psiA=psiA+C(t)*besselh(nL(i), 1, k0*r_p).*exp(sqrt(-1)*nL(i)*theta_p);
                psiB=psiB+s0*sqrt(-1)^valley_index*C(t)*besselh(nL(i)+valley_index, 1, k0*r_p).*exp(sqrt(-1)*(nL(i)+valley_index)*theta_p);

                t=t+1;
                index_p=index_p+1;
            end
        end
    end

    psiA=psiA+besselh(0,1,abs(zz)*k0)/sqrt(2);
    psiB=psiB+s0*sqrt(-1)*besselh(1,1,abs(zz)*k0).*exp(sqrt(-1)*angle(zz))/sqrt(2);
    norm_sca=sum(sum(abs(psiA).^2+abs(psiB).^2))/N^2;
    psiA=psiA./sqrt(norm_sca);
    psiB=psiB./sqrt(norm_sca);

    psiA_target=exp(sqrt(-1)*k0*xx)/sqrt(2);
    psiB_target=s0*exp(sqrt(-1)*k0*xx)/sqrt(2);
    norm_target=sum(sum(abs(psiA_target).^2+abs(psiB_target).^2))/N^2;
    psiA_target=psiA_target./sqrt(norm_target);
    psiB_target=psiB_target./sqrt(norm_target);

    score=sum(sum(abs(psiA_target-psiA).^2+abs(psiB_target-psiB).^2));
    score=score/N^2;
end

