% Loading the spam database
load spambase.data
X = spambase(:,1:57); %extracting the features into X
y = spambase(:,58);   %extracting the classes into y
y=y+1;
rng(1);
tree=fitctree(X,y)
%view(tree,'Mode','graph')

classification_error=cvloss(tree)


tree2 = prune(tree,'level',10)
view(tree2,'Mode','graph');
classification_error2=cvloss(tree2)

m=max(tree.PruneList)-1
[classification_errors,~,~,bestLevel] = cvloss(tree,'SubTrees',0:m,'KFold',10)


bestpruned=prune(tree,'Level',bestLevel);
view(bestpruned,'Mode','graph')

