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
classdef LearnAlgo_RT < LearnAlgo
    
    properties 
       name
       color
       inverseNum
       available %are we ready?
       learnedParam
    end
    
    properties (Constant)
        type = 'RT'
    end
    
    methods
        function obj = LearnAlgo_RT(p) 
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
            
           if ~isfield(p,'name')
               obj.name = 'RT';
           else
               obj.name = p.name;
           end
           
           if ~isfield(p,'color')
                obj.color = 'y';
           else
               obj.color = p.color;
           end
           
           if ~isfield(p,'inverseNum')
               obj.inverseNum = 30;
           else
               obj.inverseNum = p.inverseNum;
           end
           
           check(obj);
        end
        
        function bool = check(obj)
           % bool = check(obj)
           % Checks if all dependencies are satisfied
           %
           bool = isempty(strfind(javaclasspath,'weka'))>0;          
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
            
            [d,n] = size(X);%d表示特征的维度，n表示样本的数量
            X_tmp = X;
            
            idxa = double(idxa);
            idxb = double(idxb);
            
            [SimIndex,DisIndex] = createPair_2(X,idxa,idxb,matches);
            X_Sim = X(:,SimIndex(:,1))-X(:,SimIndex(:,2));
            Y_Sim = ones(1,size(X_Sim,2));
            X_Dis = X(:,DisIndex(:,1))-X(:,DisIndex(:,2));
            Y_Dis = zeros(1,size(X_Dis,2));
            Y_Dis(1:size(Y_Dis,2)) = -1;
            
            X_train = [X_Sim X_Dis];
            Y_train = [Y_Sim Y_Dis];
            train_data = Data;
            train_data.X = X_train';
            train_data.Y = Y_train';
            
            
            tree = weka.classifiers.trees.RandomForest();

            tree_param=wekaArgumentString({'-I','40','-K',d,'-S','1'});

            tree.setOptions(tree_param);

            weka_data = wekaCategoricalData(train_data);
            tree.buildClassifier(weka_data);
            
            covMatches    = SOPD(X_tmp,SimIndex(:,1),SimIndex(:,2)) / size(SimIndex,1);
            s.M =  0.5*(inv(covMatches));

            
            s.t = 0;
            s.learnAlgo = obj;
            s.roccolor = obj.color;
            s.tree = tree;
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
%             d = zeros(1,length(idxa));
%             for c=1:length(idxa)
%                 d(c) = (s.L*X(:,idxa(c))-X(:,idxb(c)))'*(s.L*X(:,idxa(c))-X(:,idxb(c)));
%             end
%             %d = cdistM(s.M,X,idxa,idxb); 
%             
%         end
        
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
            [da, pn] = size(Xa);
            [db, qn] = size(Xb);
            
            dist = zeros(pn,qn);
            if pn == 0 || qn == 0 || da ~= db
              return
            end
            
            tree = obj.learnedParam.tree;
            for i = 1:pn
                for j = 1:qn
                    fprintf('test data，i：%d  j:%d\n ',i,j);
                    Xij = Xa(:,i)-Xb(:,j);
                   % weka_inst = weka.core.Instance(1.0, [Xij' 1]);
                    test_data = Data;
                    test_data.X = Xij';
                   test_data.Y = 1;
                    dw = wekaCategoricalData(test_data);
                    rel = tree.distributionForInstance(dw.instance(0))';
                    p_dis = log(rel(1));
                    p_sim = Xij'*obj.learnedParam.M*Xij;
                    dist(i,j) = p_dis + p_sim;
                end
            end
        end
    end    
end

function pairIndex = createPairWise(X,idxa,idxb,matches,inverseNum)
    idxa = idxa(matches);
    idxb = idxb(matches);

   % inverseNum = 10;
    pairIndex = zeros(length(idxa)*inverseNum,3);
    for i = 1:length(idxa)
        
        pairIndex((i-1)*inverseNum+1:i*inverseNum,1) = idxa(i);
        pairIndex((i-1)*inverseNum+1:i*inverseNum,2) = idxb(i);
        
        useIndex = ones(1,length(idxb));
        useIndex(i) = 0;
        useXb = idxb(logical(useIndex));
        useXb = useXb(randperm(length(useXb)));
        pairIndex((i-1)*inverseNum+1:i*inverseNum,3) = useXb(1:inverseNum);
        
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
