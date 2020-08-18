function [labels,target] = LDA(training,testing)
d = training
%Mean of data set
m=mean(d,2); 
M=repmat(m,[1,200]); 
j=1; 
%Calculate the mean of each class
for i=0:5:195     
    mea(:,j)=mean(d(:,i+1:i+5),2); 
    me(:,i+1:i+5)=repmat(mea(:,j),[1,5]);    
    j=j+1; 
end 
temp=zeros(10304,10304);      
sw=zeros(10304,10304);  
%Calculate the within class scatter matrix
for i =0:5:195      
    temp=(d(:,i+1:i+5)-me(:,i+1:i+5))*((d(:,i+1:i+5)-me(:,i+1:i+5))');      
    sw=temp+sw;   
end;   
temp1=zeros(10304,10304);   
sb=zeros(10304,10304);  
%Calculate the between class scatter matrix
for i=1:40      
    temp1=(mea(:,i)-m)*((mea(:,i)-m)');      
    sb=temp1+sb;   
end;

Va=cov(((d-M)')); 
%PCA eigenvectors and eigenvalues 
[PCAV,PCAD]=eig(Va,'vector');  
PCAVk=PCAV(:,(10304-159:10304));    
%PCA Eigen Space selection
%Within class scatter matrix and between class scatter matrix
%projecting into PCA
swproj=PCAVk'*sw*PCAVk; 
sbproj=PCAVk'*sb*PCAVk;
%Eigenvector and Eigenvalues from between class and within class scatter matrix 
[V,D]=eig(sbproj,swproj,'vector');
%LDA eigenspace 
Proj=PCAVk*V; Projk=Proj(:,1:39);   
%Project training data onto Eigenspace
trainpro=Projk'*(d-M); 

%Testing data
testD = testing
%Find mean of testing data
testm=mean(testD,2); 
testM=repmat(testm,[1,200]);
%Project testing data into LDA space 
testpro=Projk'*(testD-testM); 
diff=pdist2(trainpro',testpro'); 
%Finding Eucledian Distances betweern Train and Test 
norm=max(diff(:)); 
labels=1/norm*(diff); 
for i=1:200    
    for j=1:200        
        if(labels(i,j)>0.50)             
            labels(i,j)=1;  %% Setting Threshold%%         
        else
            labels(i,j)=0;         
        end;     
    end; 
end; 

%Create targets
tar=[zeros(1,5),ones(1,195)]; 
tar=[tar;tar;tar;tar;tar]; 
target=zeros(200,200); 
target(1:5,:)=tar; 
for i=5:5:195 
    target(i+1:i+5,:)=circshift(tar,i,2);  
end
end
