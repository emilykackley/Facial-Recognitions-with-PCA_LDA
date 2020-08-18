function [D,results]=PCA(training,testing)
%Calculate the mean of the data matrix
m = mean(training,2);

%Subtract the mean from each image
d = training-repmat(m,1,200);

%Compute the covariance matrix
co = d*d';

%Calculate the eigenvalues and eigenvectors of the covariance matrix
[eigenvectors, eigenvalues] = eig(co);

%Sort the eigenvectors by eigenvalues
eigenvalues = diag(eigenvalues);
[temp,index] = sort(eigenvalues,'descend');

%Compute the number of eigenvalues greater than zero
c1 = 0;
for i = 1: size(eigenvalues,1)
    if(eigenvalues(i)>0)
        c1 = c1 + 1;
    end
end

%Use the eigenvectors that the corresponding eigenvalues that are greater
%than zero (this threshold can be changed to any value you want)
vec = eigenvectors(:,index(1:200));

%Projecting the training data
project_trainimg = vec'*d;

%Subtract the mean from each testing image
testing = testing-repmat(mean(testing,2),1,200);

%Project test images
project_testimg = vec'*testing;

%Euclidean distance 
D = pdist2(project_trainimg',project_testimg','Euclidean');

%results (determining what was correctly classified and what was not)
results = zeros(200,200);
for i = 1: 200
    for k = 1: 200
        if(fix((i-1)/10)==fix((k-1)/10))
            results(i,k)=0
        else
            results(i,k)=1;
        end
    end
end

tar=[zeros(1,5),ones(1,195)]; 
tar=[tar;tar;tar;tar;tar]; 
target=zeros(200,200); 
target(1:5,:)=tar; 
for i=5:5:195 
    target(i+1:i+5,:)=circshift(tar,i,2);  %% Creating Targets for each subject%% 
end
end