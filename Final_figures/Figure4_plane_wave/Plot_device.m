clear
clc

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