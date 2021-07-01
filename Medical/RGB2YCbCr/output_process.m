clc;
clear all;
 for i=1:300
      I_result=double(imread(strcat('Medical\Result\Y\',num2str(i),'.png')));  
      I_init_vi=double(imread(strcat('Medical\Function\RGB\',num2str(i),'.png')));
      
      [Y,Cb,Cr]=RGB2YCbCr(I_init_vi);
       
      I_final_YCbCr=cat(3,I_result,Cb,Cr);
      
      I_final_RGB=YCbCr2RGB(I_final_YCbCr);

      imwrite(uint8(I_final_RGB), strcat('Medical\Result\RGB\',num2str(i),'.png')); 

    end
