% ģʽʶ����ҵ��PCA����ʶ��
% ����:�㶫��ҵ��ѧ��Ϣ����ѧԺ18����ط��
% �ο���������: https://blog.csdn.net/hesays/article/details/39498375
% ����:2021/11/20
clear;
close all;
imgdata = [];%ѵ��ͼ�����
train_num = 8;%���Լ���Ŀ
%ѡ��ǰ8����Ϊѵ��������ÿһ��ѵ��ͼ������ݰ������У�������Ϊ����
%��ôÿһ��ͼ��ĺ�2��ͼ�����ݼ�Ϊ���Լ�
for i = 1:40
  for j = 1:train_num %ѡ��ǰ8����Ϊѵ����
    a = imread(strcat('D:\zeng_\Desktop\ģʽʶ����ҵ\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'));
    % imshow(a);
    b = a(1:112*92); % ��a������˳��תΪ������b
    b = double(b);
    imgdata = [imgdata; b]; % imgdata ��һ��M * N ����imgdata��ÿһ������һ��ͼƬ��M��320
  end;
end;
imgdata = imgdata'; %������ת�ã�ÿһ��Ϊһ��ͼƬ����40*8=320��
average_face = mean(imgdata,2); % ��imgdata��ÿһ�����ֵ,�洢Ϊ10304*1�ľ���

% ѵ�����е�ƽ������ʾ,��ע��
% Average_face = reshape(average_face,112,92);
% imshow(Average_face,[]);
% title(strcat('40*8��ѵ�����е�ƽ����'))

% imgdata��ÿһ�н������ֵ��������ȥ��һ�еľ�ֵ
immin = zeros(112*92,40*train_num);
for i = 1:train_num*40 
  immin(:,i) = imgdata(:,i)-average_face; % immin��һ��N*M������ѵ��ͼ��ƽ��ͼ֮��Ĳ�ֵ
end;

%covx = immin * immin';  % N * N ��Э������� 10304*10304 �Ѿ������һ�Σ���Ϊ320*320
covx = immin' * immin;  % N * N ��Э�������,��Ϊ 320*320

% %��Щ����Ϊ���������ʱ��10304*10304���õ��ľ���֮��Ϊ�˷���������320*320
% load covx;
% load V;
% load D;
% load vsort;
% load dsort;
% ��Э�������covx������ֵ����������
% �Ѿ������һ��
[V,D] = eig(covx);%VΪ������ϵ�M*M������������DΪ�ԽǾ��󣬶Խ����ϵ�ֵΪ����ֵ
d1 = diag(D);%����������ȡ��
[D_sort,index] = sort(d1);%�����ս�������
%��ʵeig��������������ֵ����������ʱ�򣬻�Ĭ�ϰ�����ֵ���մ�С�����˳������
%��ʱ���Բ鿴index���ᷢ�ְ��ս������еĶ�Ӧ����ֵ���Ǵ�1��320
cols = size(V,2);%�����������������

%�������������ն�Ӧ����ֵ�Ĵ�С�����ս�������
for i=1:cols    
    vsort(:,i) = V(:, index(cols-i+1) ); % vsort ������ǰ��������е���������,ÿһ�й���һ����������      
    dsort(i)   = d1( index(cols-i+1) );  % dsort ������ǰ��������е�����ֵ����һά������ 
end 

%����ѡ����90%������ 
sumEng = sum(dsort); %������
sumEngReal = 0;%Ŀǰ������
p = 0; %ά��
%ѡ�񹹳�90%������������ֵ
while(sumEngReal/sumEng<0.9)       %���ѡ���ά�ȵ���������90%
    p=p+1;                         %��ά�ȼ�һ
    sumEngReal = sum(dsort(1:p));  %���ٴμ����ʱ������
end
a=1:1:cols;
for i=1:1:cols
    y(i)=sum(dsort(a(1:i)) );
end

figure(1)
y1=ones(1,cols);
plot(a,y/sumEng,a,y1*0.9,'linewidth',1);
grid
title(sprintf('�����������Ҫ%dά��',p));
xlabel('ǰn������ֵ');
ylabel('��ռ��������ռ�ٷֱ�');
hold on;

%��ʾ������,���ù�һ�Σ���ͼ������Ϊ�˷���������Լ���Э������󣬴˶α�ע��
% vsort_reduce = vsort(:,1:6);
% for j = 1:6
%     subplot(2,3,j)
%     imshow(reshape(vsort_reduce(:,j),112,92),[]);
% end

% ѵ���õ�����������ϵ
i=1;
while (i<=p && dsort(i)>0)
  base(:,i) = dsort(i)^(-1/2)*immin * vsort(:,i); % base��N��p�׾�����������ͶӰ������dsort(i)^(1/2)�Ƕ�����ͼ��ı�׼��
  i = i + 1;
end

reference = base'*immin;% ��ѵ������������ϵ�Ͻ���ͶӰ,�õ�һ�� p*M �׾���Ϊ�ο�
accu = 0; %����׼ȷ��
erroNum = 0; %ʶ�������

% ���Թ���
% ʹ��K���ڽ��㷨(����ȡK =1,��ʹ����С����)
% ���� K=1����ȷ��Ϊ96.25%;K=2,3,4����ȷ��Ϊ92.50%;K=5����ȷ��Ϊ91.25%,�ʽ�������
KNN_RANK = 1;
for i=1:40
   for j=9:10 %����40 x 2 ������ͼ��
      a=imread(strcat('D:\zeng_\Desktop\ģʽʶ����ҵ\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'));
      b=a(1:10304);
      b=double(b);
      b=b';       
      
      object = base'*(b-average_face);

      temp = zeros(1,320);
      CNT_MATRIX = zeros(1,40); %��������
      for k= 1:320
          temp(:,k)= norm(object - reference(:,k));
      end;
      [distance_sotr,INDEX] = sort(temp); %��������,��ȡ����Ӧ������ֵINDEX
      
      %������ѡ��Kȡ1
      for inx = 1:KNN_RANK
          temp_which = INDEX(inx)/8;  %�����С��3���ٽ�����������ֵ��Ӧ�����Ǹ�����
          which = ceil(temp_which);%������������ȡ������INDEX��i����1��8�������ڵ�һ����
          CNT_MATRIX(1,which) = CNT_MATRIX(1,which) + 1;%�ڼ��������м���
      end;
      
      [KNN_sort,KNN_result] = sort(CNT_MATRIX,'descend'); %���������󰴽������в��ҵ�ȡ������ֵ
      
      if i == KNN_result(1)
          accu = accu+1;
      else %���Ƴ�ʶ������ͼ��
          erroNum = erroNum+1;
          figure(erroNum+1)
          subplot(2,1,1);
          imshow(strcat('D:\zeng_\Desktop\ģʽʶ����ҵ\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'))
          title(sprintf('ʶ�����ĵ�%d��ͼƬ,���ڵ�%d�˵�%d�ţ�ʶ��Ϊ��%d��',erroNum,i,j,KNN_result(1)));
          subplot(2,1,2)
          imshow(strcat('D:\zeng_\Desktop\ģʽʶ����ҵ\ORL2\ORL_Faces\s',num2str(KNN_result(1)),'\1.pgm'))
          title(sprintf('���еĵ�%d�˵�ͼ��Ϊ',KNN_result(1)));
      end;
   end;
end;

accuracy = accu / 80 %�����ȷ��