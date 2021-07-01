clc;
clear all;
 for i=1:300
      I_result=double(imread(strcat('MFF\Result\Y\',num2str(i),'.png')));        
      I_init_under=double(imread(strcat('MFF\Far\RGB\',num2str(i),'.png')));
      I_init_over=double(imread(strcat('MFF\Near\RGB\',num2str(i),'.png')));     
      [Y1,Cb1,Cr1]=RGB2YCbCr(I_init_under);
      [Y2,Cb2,Cr2]=RGB2YCbCr(I_init_over);
      
      [H,W]=size(Cb1);
      Cb=ones([H,W]);
      Cr=ones([H,W]);     
      for k=1:H
          for n=1:W
           if (abs(Cb1(k,n)-128)==0&&abs(Cb2(k,n)-128)==0)  
              Cb(k,n)=128;
           else
                middle_1= Cb1(k,n)*abs(Cb1(k,n)-128)+Cb2(k,n)*abs(Cb2(k,n)-128);
                middle_2=abs(Cb1(k,n)-128)+abs(Cb2(k,n)-128);
                Cb(k,n)=middle_1/middle_2;
           end   
            if (abs(Cr1(k,n)-128)==0&&abs(Cr2(k,n)-128)==0)      
               Cr(k,n)=128;  
            else
                middle_3=Cr1(k,n)*abs(Cr1(k,n)-128)+Cr2(k,n)*abs(Cr2(k,n)-128);
                middle_4=abs(Cr1(k,n)-128)+abs(Cr2(k,n)-128); 
                Cr(k,n)=middle_3/middle_4;
            end            
          end
      end    
      I_final_YCbCr=cat(3,I_result,Cb,Cr);
      I_final_RGB=YCbCr2RGB(I_final_YCbCr);
      imwrite(uint8(I_final_RGB), strcat('MFF\Result\RGB\',num2str(i),'.png')); 
  end
   
      

