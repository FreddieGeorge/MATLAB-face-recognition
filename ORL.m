% 模式识别作业：PCA人脸识别
% 作者:广东工业大学信息工程学院18级曾胤宁
% 参考代码链接: https://blog.csdn.net/hesays/article/details/39498375
% 日期:2021/11/20
clear;
close all;
imgdata = [];%训练图像矩阵
train_num = 8;%测试集数目
%选择前8个作为训练集，将每一个训练图像的数据按列排列，并保存为矩阵
%那么每一组图像的后2张图像数据即为测试集
for i = 1:40
  for j = 1:train_num %选择前8个作为训练集
    a = imread(strcat('D:\zeng_\Desktop\模式识别作业\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'));
    % imshow(a);
    b = a(1:112*92); % 将a矩阵按列顺序转为行向量b
    b = double(b);
    imgdata = [imgdata; b]; % imgdata 是一个M * N 矩阵，imgdata中每一行数据一张图片，M＝320
  end;
end;
imgdata = imgdata'; %将矩阵转置，每一列为一张图片，共40*8=320列
average_face = mean(imgdata,2); % 对imgdata的每一行求均值,存储为10304*1的矩阵

% 训练集中的平均脸显示,可注释
% Average_face = reshape(average_face,112,92);
% imshow(Average_face,[]);
% title(strcat('40*8张训练集中的平均脸'))

% imgdata的每一行进行零均值化，即减去这一行的均值
immin = zeros(112*92,40*train_num);
for i = 1:train_num*40 
  immin(:,i) = imgdata(:,i)-average_face; % immin是一个N*M矩阵，是训练图和平均图之间的差值
end;

%covx = immin * immin';  % N * N 阶协方差矩阵 10304*10304 已经计算过一次，简化为320*320
covx = immin' * immin;  % N * N 阶协方差矩阵,简化为 320*320

% %这些数据为输出本征脸时的10304*10304所用到的矩阵，之后为了方便计算采用320*320
% load covx;
% load V;
% load D;
% load vsort;
% load dsort;
% 对协方差矩阵covx求特征值和特征向量
% 已经计算过一次
[V,D] = eig(covx);%V为按列组合的M*M的特征向量，D为对角矩阵，对角线上的值为特征值
d1 = diag(D);%将特征向量取出
[D_sort,index] = sort(d1);%并按照降序排列
%其实eig函数在生成特征值与特征向量时候，会默认把特征值按照从小到大的顺序排列
%此时可以查看index，会发现按照降序排列的对应索引值就是从1到320
cols = size(V,2);%特征向量矩阵的列数

%将特征向量按照对应特征值的大小排序按照降序排列
for i=1:cols    
    vsort(:,i) = V(:, index(cols-i+1) ); % vsort 保存的是按降序排列的特征向量,每一列构成一个特征向量      
    dsort(i)   = d1( index(cols-i+1) );  % dsort 保存的是按降序排列的特征值，是一维行向量 
end 

%以下选择保留90%的能量 
sumEng = sum(dsort); %总能量
sumEngReal = 0;%目前的能量
p = 0; %维数
%选择构成90%的能量的特征值
while(sumEngReal/sumEng<0.9)       %如果选择的维度的能量不足90%
    p=p+1;                         %则将维度加一
    sumEngReal = sum(dsort(1:p));  %并再次计算此时的能量
end
a=1:1:cols;
for i=1:1:cols
    y(i)=sum(dsort(a(1:i)) );
end

figure(1)
y1=ones(1,cols);
plot(a,y/sumEng,a,y1*0.9,'linewidth',1);
grid
title(sprintf('计算出最少需要%d维度',p));
xlabel('前n个特征值');
ylabel('所占总能量的占百分比');
hold on;

%显示本征脸,调用过一次，截图完结果后为了方便后续调试简化了协方差矩阵，此段便注释
% vsort_reduce = vsort(:,1:6);
% for j = 1:6
%     subplot(2,3,j)
%     imshow(reshape(vsort_reduce(:,j),112,92),[]);
% end

% 训练得到特征脸坐标系
i=1;
while (i<=p && dsort(i)>0)
  base(:,i) = dsort(i)^(-1/2)*immin * vsort(:,i); % base是N×p阶矩阵，用来进行投影，除以dsort(i)^(1/2)是对人脸图像的标准化
  i = i + 1;
end

reference = base'*immin;% 将训练样本对坐标系上进行投影,得到一个 p*M 阶矩阵为参考
accu = 0; %计算准确度
erroNum = 0; %识别错误数

% 测试过程
% 使用K最邻近算法(这里取K =1,即使用最小距离)
% 测试 K=1，精确度为96.25%;K=2,3,4，精确度为92.50%;K=5，精确度为91.25%,呈降低趋势
KNN_RANK = 1;
for i=1:40
   for j=9:10 %读入40 x 2 副测试图像
      a=imread(strcat('D:\zeng_\Desktop\模式识别作业\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'));
      b=a(1:10304);
      b=double(b);
      b=b';       
      
      object = base'*(b-average_face);

      temp = zeros(1,320);
      CNT_MATRIX = zeros(1,40); %计数矩阵
      for k= 1:320
          temp(:,k)= norm(object - reference(:,k));
      end;
      [distance_sotr,INDEX] = sort(temp); %降序排序,并取出对应的索引值INDEX
      
      %这里我选择K取1
      for inx = 1:KNN_RANK
          temp_which = INDEX(inx)/8;  %求出最小的3个临近样本的索引值对应的是那个样本
          which = ceil(temp_which);%并且向正无穷取整，如INDEX（i）在1到8，则属于第一个人
          CNT_MATRIX(1,which) = CNT_MATRIX(1,which) + 1;%在计数矩阵中计数
      end;
      
      [KNN_sort,KNN_result] = sort(CNT_MATRIX,'descend'); %将计数矩阵按降序排列并找到取出索引值
      
      if i == KNN_result(1)
          accu = accu+1;
      else %绘制出识别错误的图像
          erroNum = erroNum+1;
          figure(erroNum+1)
          subplot(2,1,1);
          imshow(strcat('D:\zeng_\Desktop\模式识别作业\ORL2\ORL_Faces\s',num2str(i),'\',num2str(j),'.pgm'))
          title(sprintf('识别错误的第%d张图片,属于第%d人第%d张，识别为第%d人',erroNum,i,j,KNN_result(1)));
          subplot(2,1,2)
          imshow(strcat('D:\zeng_\Desktop\模式识别作业\ORL2\ORL_Faces\s',num2str(KNN_result(1)),'\1.pgm'))
          title(sprintf('误判的第%d人的图像为',KNN_result(1)));
      end;
   end;
end;

accuracy = accu / 80 %输出精确度