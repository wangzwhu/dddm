function [result_sparse,result_knn,result ] = calcMCMC2( M, data, idxa, idxb, idxtrain, idxtest )

KNN = 24;     % 32 48
alph = 0.5;   
beta = 1;  

rho=0.1;
opts=[];
opts.init=2;
opts.tFlag=5;
opts.maxIter=100;
opts.nFlag=0;
opts.rFlag=1;
opts.rsL2=0;

XATrain = data(:,idxa(idxtrain));
XBTrain = data(:,idxb(idxtrain));
XATest = data(:,idxa(idxtest));
XBTest = data(:,idxb(idxtest));

sparse_presentation_a = [];
sparse_presentation_b = [];
for i = 1 : size(idxtest,2)
    y = XBTest(:, i);
    [x, funVal]= nnLeastR(XBTrain, y, rho, opts);
    sparse_presentation_b = [sparse_presentation_b  x(:)];  
    y = XATest(:, i);
    [x, funVal]= nnLeastR(XATrain, y, rho, opts);
    sparse_presentation_a = [sparse_presentation_a  x(:)];
end


dist_testa_testa_E   = sqdist(data(:,idxa(idxtest)), data(:,idxa(idxtest)));
dist_traina_traina_E = sqdist(data(:,idxa(idxtrain)), data(:,idxa(idxtrain)));
dist_testb_testb_E   = sqdist(data(:,idxb(idxtest)), data(:,idxb(idxtest)));
dist_trainb_trainb_E = sqdist(data(:,idxb(idxtrain)), data(:,idxb(idxtrain)));
dist_trainb_testb_E  = sqdist(data(:,idxb(idxtrain)), data(:,idxb(idxtest)));
dist_testb_trainb_E  = sqdist(data(:,idxb(idxtest)), data(:,idxb(idxtrain)));
dist_testa_traina_E  = sqdist(data(:,idxa(idxtest)), data(:,idxa(idxtrain)));
dist_traina_testa_E  = sqdist(data(:,idxa(idxtrain)), data(:,idxa(idxtest)));
dist_traina_trainb_M = sqdist(data(:,idxa(idxtrain)), data(:,idxb(idxtrain)),M);
dist_testa_testb_M   = sqdist(data(:,idxa(idxtest)), data(:,idxb(idxtest)),M);
dist_traina_testb_M  = sqdist(data(:,idxa(idxtrain)), data(:,idxb(idxtest)),M);
dist_testa_trainb_M  = sqdist(data(:,idxa(idxtest)), data(:,idxb(idxtrain)),M);
dist_testb_traina_M  = sqdist(data(:,idxb(idxtest)), data(:,idxa(idxtrain)),M);

dist_final = [];
dist_final_sparse = [];
dist_final_knn = [];
dist_final_combine = [];

rank_knn = [];
rank_sparse = [];
rank = [];

for i = 1:size(dist_testa_testb_M,2)
    dist_final(i,:) = dist_testa_testb_M(i,:);
    
    [tmp, neighbor] = sort(dist_testa_trainb_M(i,:),'ascend');
    for j = 1:size(dist_testb_trainb_E,2)
        [tmp,neighbor2] = sort(dist_testb_trainb_E(j,:),'ascend');
        neighbor_intersection = intersect(neighbor(1:KNN), neighbor2(1:KNN));
        sim_b(i, j) = length(neighbor_intersection);
    end
    
    [tmp, neighbor] = sort(dist_testa_traina_E(i,:),'ascend');
    for j = 1:size(dist_testb_traina_M,2)
        [tmp,neighbor2] = sort(dist_testb_traina_M(j,:),'ascend');
        neighbor_intersection = intersect(neighbor(1:KNN), neighbor2(1:KNN));
        sim_a(i, j) = length(neighbor_intersection);
    end
    
    present_a = sparse_presentation_a(:, i);
    non_zero_a = find(present_a~=0);
    for j = 1:size(sparse_presentation_b,2)
        present_b = sparse_presentation_b(:, j);
        non_zero_b = find(present_b~=0);
        present_intersection = intersect(non_zero_a, non_zero_b);
        sim_s(i, j) = length(present_intersection);
    end
end 

fs = 1./(1+sim_s);
fp = 1./(1+sim_b+sim_a);
fs = power(fs, alph);
fp = power(fp, beta);
dist_final_sparse  = dist_final.*fs;
dist_final_knn     = dist_final.*fp;
dist_final_combine = dist_final.*fs.*fp;

for i = 1:size(dist_testa_testb_M,2)
    [tmp_knn,   rank_knn(i,:)]    = sort(dist_final_knn(i,:),'ascend');
    [tmp_sparse,rank_sparse(i,:)] = sort(dist_final_sparse(i,:),'ascend');
    [tmp,       rank(i,:)]        = sort(dist_final_combine(i,:),'ascend');
end    

result_knn = zeros(1,size(dist_testa_testb_M,2));
result_sparse = zeros(1,size(dist_testa_testb_M,2));
result = zeros(1,size(dist_testa_testb_M,2));

for pairCounter=1:size(dist_testa_testb_M,2)
    idx_knn  = rank_knn(pairCounter,:);
    idx_sparse = rank_sparse(pairCounter,:);
    idx = rank(pairCounter,:);
    
    result_knn(idx_knn==pairCounter) = result_knn(idx_knn==pairCounter) + 1;
    result_sparse(idx_sparse==pairCounter) = result_sparse(idx_sparse==pairCounter) + 1;
    result(idx==pairCounter) = result(idx==pairCounter) + 1;
end


tmp_knn = 0;
tmp_sparse = 0;
tmp = 0;

for counter=1:length(result)
    result(counter) = result(counter) + tmp;
    result_sparse(counter) = result_sparse(counter) + tmp_sparse;
    result_knn(counter) = result_knn(counter) + tmp_knn;
    
    tmp = result(counter);
    tmp_sparse = result_sparse(counter);
    tmp_knn = result_knn(counter);
end




