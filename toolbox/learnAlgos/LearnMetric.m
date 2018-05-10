function [M] = LearnMetric( X,idxa,idxb,matches,method )
%LEARNMETRIC Summary of this function goes here
%   Detailed explanation goes here
    switch method
        case 'KISSME' 
            [M] = learnMetricKissMe(X,idxa,idxb,matches);
        case 'Mahal' 
            [M] = learnMetricMahal(X,idxa,idxb,matches);
        case 'ITML'
            M = PairMetricLearning(@ItmlAlg, idxa', idxb', matches, X');
        otherwise
            M = eye(size(X,1));
    end
end

function [M] =  learnMetricKissMe(X,idxa,idxb,matches)

%             Eqn. (12) - sum of outer products of pairwise differences (similar pairs)
%             normalized by the number of similar pairs.
             covMatches    = SOPD(X,idxa(matches),idxb(matches)) / sum(matches);
%             Eqn. (13) - sum of outer products of pairwise differences (dissimilar pairs)
%             normalized by the number of dissimilar pairs.
             covNonMatches = SOPD(X,idxa(~matches),idxb(~matches)) / sum(~matches);
%             
%             Eqn. (15-16)
            M = inv(covMatches) - inv(covNonMatches);   %inv 求逆矩阵
            M = validateCovMatrix(M);%得到矩阵的半正定矩阵

end

function [M] =  learnMetricMahal(X,idxa,idxb,matches)

     covMatches = SOPD(X,idxa(matches),idxb(matches)) / sum(matches);
     M = inv(covMatches); 
end


