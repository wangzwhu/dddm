function [ L1,error ] = LPMatrix( X,PairIndex,LPM_Param)
%LPMATRIX Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 2
        return;
    end
    [d,n] = size(X);%d表示特征的维度，n表示样本的数量
    if nargin<3
       LPM_Param = struct();
    end
    
    if ~isfield(LPM_Param,'lambda')
         LPM_Param.lambda =  0.005;
    end
           
    if ~isfield(LPM_Param,'MaxIter')
        LPM_Param.MaxIter = 1000;
    end
    
    if ~isfield(LPM_Param,'M')
        LPM_Param.M = eye(d);
    end
    
    if ~isfield(LPM_Param,'L2')
        LPM_Param.L2 = eye(d);
    end
    
    if ~isfield(LPM_Param,'L1')
        LPM_Param.L1 = eye(d);
    end

    pairNum = size(PairIndex,1);
    M = LPM_Param.M;
    L1 = LPM_Param.L1;
    L2 = LPM_Param.L2;
    lamda = LPM_Param.lambda;
    %计算样本的代价函数
    error_last = 0;
    for i=1:pairNum
        error_tmp = onePairError(X(:,PairIndex(i,1)),X(:,PairIndex(i,2)),X(:,PairIndex(i,3)),M,L1,L2);
        error_last = error_last + log(1+exp(error_tmp));
    end
    disp(sprintf('the Sample total error: %f',error_last));


    L_last = L1;
    for iterCount = 1:LPM_Param.MaxIter
       
        %计算梯度
        gradient = zeros(d);
        for i=1:pairNum
            direct_tmp = 2*M*LPM_Param.L2*(X(:,PairIndex(i,3))-X(:,PairIndex(i,2)))*(X(:,PairIndex(i,1))');
            
            error_tmp = onePairError(X(:,PairIndex(i,1)),X(:,PairIndex(i,2)),X(:,PairIndex(i,3)),M,L1,L2); 
            
            gradient = gradient+1/(1+exp(-error_tmp))*direct_tmp;
        end  
        
        done = 0;
        findLamdaIter = 10;
        error_total=0;
        
        while (~done) & (findLamdaIter>0)
            findLamdaIter = findLamdaIter - 1;
            L1 = L_last-lamda*gradient;

            error_total=0;              
            for i=1:pairNum
                error_tmp = onePairError(X(:,PairIndex(i,1)),X(:,PairIndex(i,2)),X(:,PairIndex(i,3)),M,L1,L2); 
                error_total = error_total + log(1+exp(error_tmp));
            end
            %disp(sprintf('find Lamda the %d Iter, Lamda:%f, Last error:%f,error:%f',(10-findLamdaIter),lamda,error_last,error_total));
            if(error_total>=error_last)
                lamda = lamda/2;
            else                        
                lamda = lamda*1.05;
                done = 1;
                L_last = L1;
            end
        end
        
       % disp(sprintf('the %d Iter, Lamda:%f, Last error:%f,error:%f',iterCount,lamda,error_last,error_total));
        error = error_last;
        L1 = L_last;
        
        if (error_total>=error_last) 
            disp(sprintf('No better error. the %d Iter, Lamda:%f, error:%f',iterCount,lamda,error_total));
            break;
        elseif((error_last -error_total)<1e-5)
            disp(sprintf('error no change. last_error:%f, error_tmp%f',error_last,error_total));
            break;
        else
            error_last = error_total;
        end
    end

end

function error = onePairError(Xi,Xj,Xn,M,L1,L2)
    dif_s = L1*Xi-L2*Xj;
    dis_s = dif_s'*M*dif_s;

    dif_d = L1*Xi-L2*Xn;
    dis_d = dif_d'*M*dif_d;
    error = dis_s - dis_d;
end