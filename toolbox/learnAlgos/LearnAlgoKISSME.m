% LEARNALGOKISSME Keep it simple and straightforward metric learning!
%
% LearnAlgoKISSME properties:
%   p  - Parameter struct.
%   p.roccolor - default 'g'.
%   p.pmetric  - Backproject on the cone of p.s.d. matrices? Default 1.
%   p.lambda   - Influence of the dissimilar pairs. Range 0 to 1, default
%   1.
%   s - struct to store the last result
%   available - if the LearnAlgoKISSME is ready
%
% LearnAlgoKISSME properties (Constant):
%   type - string identifier default 'kissme'
%
% LearnAlgoKISSME methods:
%   LearnAlgoKISSME  - ctor.
%   learnPairwise    - learn from pairwise equivalence labels
%   learn            - learn with fully supervised data
%   dist             - calculate distance between pairs of samples
%
% See also LearnAlgoLMNN,LearnAlgoITML,LearnAlgoLDML
%
% copyright by Martin Koestinger (2011)
% Graz University of Technology
% contact koestinger@icg.tugraz.at
%
% For more information, see <a href="matlab: 
% web('http://lrs.icg.tugraz.at/members/koestinger')">the ICG Web site</a>.
%
classdef LearnAlgoKISSME < LearnAlgo
    
    properties 
       p %parameters 
       s %struct
       available %are we ready?
       name
       color
       inverseNum
       learnedParam
       param
    end
    
    properties (Constant)
        type = 'kissme'
    end
    
    methods
        function obj = LearnAlgoKISSME(p) 
            % obj = LearnAlgoKISSME(p) 
            % Constructor
            %
            % parameters:
            %   p  - Parameter struct.
            %   p.roccolor - default 'g'.
            %   p.pmetric - Backproject on the cone of p.s.d. matrices? Default 1.
            %   p.lambda - Influence of the dissimilar pairs. Range 0 to 1, default 1.
            %
            % return:
            %   obj       - object handle to instance of class LearnAlgoKISSME
            %   
           if nargin < 1
               p = struct(); 
           end
            
           if ~isfield(p,'lambda')
             p.lambda =  1;
           end
           
           if ~isfield(p,'roccolor')
                p.roccolor = 'g';
           end
           
           if ~isfield(p,'pmetric')
               p.pmetric = 1;
           end
           obj.name = 'kissme';
           obj.color = 'g';
           obj.p  = p;
           check(obj);
        end
        
        function bool = check(obj)
           % bool = check(obj)
           % Checks if all dependencies are satisfied
           %
           bool = exist('SOPD') ~= 0;          
           if ~bool
               className = class(obj);
               fprintf('Sorry %s not available\n',className);
           end
           obj.available = bool;
        end
        
        function s = learnPairwise(obj,X,idxa,idxb,matches)
            % s = learnPairwise(obj,X,idxa,idxb,matches)
            % Learn from pairwise equivalence labels
            %
            % parameters:
            %   obj       - instance of class LearnAlgoKISSME
            %   X         - input matrix, each column is an input vector 
            %   [DxN*2]. N is the number of pairs. D is the feature 
            %   dimensionality
            %   idxa      - index of image A in X [1xN]
            %   idxb      - index of image B in X [1xN]
            %   matches   - matches defines if a pair is similar (1) or 
            %   dissimilar (0)
            %
            % return:
            %   s         - Result data struct
            %   s.M       - Trained quadratic distance metric
            %   s.t       - Training time in seconds
            %   s.p       - Used parameters, see LearnAlgoKISSME properties for details.s
            %   s.learnAlgo - class handle to obj
            %   s.roccolor  - line color for ROC curve, default 'g'
            %   
            if ~obj.available
                s = struct();
                return;
            end
            
            idxa = double(idxa);
            idxb = double(idxb);
            
            X_tmp = X;
            [SimIndex,DisIndex] = createPair_2(X,idxa,idxb,matches);
            
            
            %--------------------------------------------------------------
            %   KISS Metric Learning CORE ALGORITHM
            %
            
            tic;
            % Eqn. (12) - sum of outer products of pairwise differences (similar pairs)
            % normalized by the number of similar pairs.
            covMatches    = SOPD(X_tmp,SimIndex(:,1),SimIndex(:,2)) / size(SimIndex,1);
            % Eqn. (13) - sum of outer products of pairwise differences (dissimilar pairs)
            % normalized by the number of dissimilar pairs.
            covNonMatches = SOPD(X_tmp,DisIndex(:,1),DisIndex(:,2)) / size(DisIndex,1);
            t = toc;
            
            tic;
            % Eqn. (15-16)
            s.M = inv(covMatches) - obj.p.lambda * inv(covNonMatches);   %inv 求逆矩阵
            if obj.p.pmetric
                % to induce a valid pseudo metric we enforce that  M is p.s.d.
                % by clipping the spectrum
                %s.M = validateCovMatrix(s.M);%得到矩阵的半正定矩阵
            end
            %s.M = -inv(covNonMatches);
            s.t = toc + t;   
            
            %
            %   END KISS Metric Learning CORE ALGORITHM
            %--------------------------------------------------------------
            
            s.learnAlgo = obj;
            s.roccolor = obj.p.roccolor;
            obj.learnedParam = s;
        end
        
        function s = learn(obj,X,y)
            % not implemented yet, sorry.
            if ~obj.available
                s = struct();
                return;
            end
            
            s.roccolor = obj.p.roccolor;
            error('not implemented yet!');
        end
        
        function dist = dist(obj, Xa, Xb)
            % d = dist(obj, s, X, idxa,idxb)
            % Calculate the distance between pairs of samples
            %
            % parameters:
            %   obj       - instance of class LearnAlgoKISSME
            %   s         - struct with member M, quadratic form
            %   X         - Input matrix (each column is an input vector)
            %   idxa,idxb - idxa(c),idxb(c) index of images a,b, pair c in X
            %   matches   - matches(c) defines if pair c is similar (1) or dissimilar (0)
            %
            % return:
            %   d         - distance for each pair specified in idxa,idxb
            %   
           % d = cdistM(s.M,X,idxa,idxb); 
            %d = sqdist(X(:,idxa), X(:,idxb),s.M);
            %d = sqdist(Xa, Xb,obj.param.M);
            dist = sqdist(Xa, Xb,obj.learnedParam.M);
            
%             [da, pn] = size(Xa);
%             [db, qn] = size(Xb);
            
%             dist = zeros(pn,qn);
%             if pn == 0 || qn == 0 || da ~= db
%               return
%             end
%             
%             for i = 1:pn
%                 for j = 1:qn
%                     Xij = Xa(:,i)-Xb(:,j);
%                     dis_a_b = Xij'*obj.learnedParam.M*Xij;
%                     Xji = Xb(:,j)-Xa(:,i);
%                     dis_b_a = Xji'*obj.learnedParam.M*Xji;
%                     
% %                     Xji = Xa(:,i)-obj.learnedParam.Lb*Xb(:,j);
% %                     dis_b_a = Xji'*obj.learnedParam.M*Xji;
%                     
%                     dist(i,j) = dis_a_b+dis_b_a;% + dis_b_a;
%                 end
%             end
        end
%         function d = dist(obj, s, X, idxa,idxb)
%             % d = dist(obj, s, X, idxa,idxb)
%             % Calculate the distance between pairs of samples
%             %
%             % parameters:
%             %   obj       - instance of class LearnAlgoKISSME
%             %   s         - struct with member M, quadratic form
%             %   X         - Input matrix (each column is an input vector)
%             %   idxa,idxb - idxa(c),idxb(c) index of images a,b, pair c in X
%             %   matches   - matches(c) defines if pair c is similar (1) or dissimilar (0)
%             %
%             % return:
%             %   d         - distance for each pair specified in idxa,idxb
%             %   
%             d = cdistM(s.M,X,idxa,idxb); 
%             %d = sqdist(X(:,idxa), X(:,idxb),s.M);
%            % d = sqdist(Xa, Xb,obj.param.M);
%         end
    end    
end

function [SimIndex,DisIndex] = createPair_2(X,idxa,idxb,matches)
    idxa = idxa(matches);
    idxb = idxb(matches);

    inverseNum = 5;
    SimIndex = zeros(length(idxa),2);
    SimIndex(:,1) = idxa;
    SimIndex(:,2) = idxb;
    
    DisIndex = zeros(length(idxa)*inverseNum,2);
    for i = 1:length(idxa)
        
        DisIndex((i-1)*inverseNum+1:i*inverseNum,1) = idxa(i);
        
        useIndex = ones(1,length(idxb));
        useIndex(i) = 0;
        useXb = idxb(logical(useIndex));
        useXb = useXb(randperm(length(useXb)));
        DisIndex((i-1)*inverseNum+1:i*inverseNum,2) = useXb(1:inverseNum);
        
    end
end