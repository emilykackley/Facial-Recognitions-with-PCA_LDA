%--------------------------------Method 1--------------------------------%
%For each subject, use the first five images (1.pgm to 5.pgm) for training 
%the subspace. Use the files 6.pgm to 10.pgm for the performance evaluation

%Create filepaths for all subjects
filepaths = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s',a,'\');
    filepaths{i}=x
end
%Create directories (string) of all subject folders
sub_dir = [];
for i = 1: 40
    a = int2str(i)
    x = strcat('\\kc.umkc.edu\kc-users\home\e\ek6w8\My Documents\Project #1\att_faces\s',a,'\*.pgm');
    x = char(x)
    x = dir(x)
    sub_dir{i}=x
end

%Create a training set made of the first 5 images for each test subject
%(200 images total)
training = cell(1,200);
%Create a training set made of the first 5 images for each test subject
%(200 images total)
testing = cell(1,200);

%Read all images into the appropriate training and testing arrays
[training,testing] = sub_images(sub_dir,filepaths)

%Convert cell arrays to matrices
training = cell2mat(training);
testing = cell2mat(testing);

[D_PCA,T_PCA] = PCA(training,testing);
[D_LDA,T_LDA] = LDA(training,testing);

[GAR_PCA,FAR_PCA,EER_PCA]=score_fusion(D_PCA,T_PCA,2,'',1);
[GAR_LDA,FAR_LDA,EER_LDA]=score_fusion(D_LDA,T_LDA,2,'',1);

%Fusion at score-level: average, min, and max
for i=1:length(GAR_LDA)
    GAR_avg(i) = (GAR_PCA(i)+GAR_LDA(i))/2;
    FAR_avg(i) = (FAR_PCA(i)+FAR_LDA(i))/2;
    GAR_max(i) = max(GAR_PCA(i),GAR_LDA(i));
    FAR_max(i) = max(FAR_PCA(i),FAR_LDA(i));
    GAR_min(i) = min(GAR_PCA(i),GAR_LDA(i));
    FAR_min(i) = min(FAR_PCA(i),FAR_LDA(i));
end
GAR_avg = GAR_avg(:);
FAR_avg = FAR_avg(:);
GAR_max = GAR_max(:);
FAR_max = FAR_max(:);
GAR_min = GAR_min(:);
FAR_min = FAR_min(:);

ind=length(GAR_PCA)

FRR_avg = 1-GAR_avg;
EER_avg = (FRR_avg(ind)+FAR_avg(ind))/2;

FRR_max = 1-GAR_max;
EER_max = (FRR_max(ind)+FAR_max(ind))/2;

FRR_min = 1-GAR_min;
EER_min = (FRR_min(ind)+FAR_min(ind))/2;

%Plot the ROC curve
figure,plot(FAR_PCA,GAR_PCA,FAR_LDA,GAR_LDA,FAR_avg,GAR_avg,FAR_max,GAR_max,FAR_min,GAR_min),axis([-0.002 1 0 1.002]),title(['ROC Curves - Score-Level Fusion']),xlabel('FAR'),ylabel('GAR'),legend('PCA (no fusion)','LDA (no fusion)','Score-Level Fusion (Average)','Score-Level Fusion (Max)','Score-Level Fusion (Min)','Location','southeast');

%Multi-Instance Fusion at score level
[GAR_PCAMI,FAR_PCAMI,EER_PCAMI] = ezroc3_MI(D_PCA,T_PCA,2,'',1)
[GAR_LDAMI,FAR_LDAMI,EER_LDAMI] = ezroc3_MI(D_LDA,T_LDA,2,'',1)

%Plot the ROC curve
figure,plot(FAR_PCA,GAR_PCA,FAR_LDA,GAR_LDA,FAR_PCAMI,GAR_PCAMI,FAR_LDAMI,GAR_LDAMI),axis([-0.002 1 0 1.002]),title(['ROC Curves - Multi-Instance Fusion']),xlabel('FAR'),ylabel('GAR'),legend('PCA (without multi-instance fusion)','LDA (without multi-instance fusion)','PCA (with multi-instance fusion)','LDA (with multi-instance fusion)','Location','southeast');

%Function to import images from att_faces folder
function [train,test] = sub_images(sub_directory,filepath)
%200 training images (first 5 from each subject)
train = cell(1,200);
%200 testing images (last 5 from each test subject)
test = cell(1,200);
%a and b are placeholders for traversing through the train and test cell
%arrays
a = 1;
b = 1;

%Loop for all 40 subjects
for k = 1: 40
    fp = filepath{k}
    x = sub_directory{k}
    directory = x
    for i = 1: 10
        if i < 6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Read image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to training set
            train{a}=temp;
            a = a+1;
        end
        if i>=6
            %Get filename
            filename = strcat(fp,directory(i).name);
            %Raed image
            temp = imread(filename);
            %Reshape image
            temp = reshape(temp,prod(size(temp)),1);
            temp = double(temp);
            %Add image to testing set
            test{b}=temp;
            b = b+1;
        end
    end
end
end