%LEARNALGOLMNN Wrapper class to the actual LMNN code
classdef LearnAlgoLMNN < LearnAlgo
    
    properties 
       p %parameters 
       s %struct
       available
       fhanlde
       name
       color
       inverseNum
       learnedParam
    end
    
    properties (Constant)
        type = 'lmnn'
    end
    
    methods
        function obj = LearnAlgoLMNN(p)
            if nargin < 1
              p = struct(); 
            end           
            if ~isfield(p,'knn')
                p.knn = 1;
            end

            if ~isfield(p,'maxiter')
                p.maxiter = 1000; %std
            end

            if ~isfield(p,'validation')
                p.validation = 0;
            end
            
            if ~isfield(p,'roccolor')
                p.roccolor = 'k';
            end
            
            if ~isfield(p,'quiet')
                p.quiet = 1;
            end
            obj.name = 'lmnn';
           obj.color = 'k';
            obj.p  = p;
            check(obj);
        end
        
        function bool = check(obj)
           bool = exist('lmnn.m') == 2;
           if ~bool
               fprintf('Sorry %s not available\n',obj.type);
           end
           obj.fhanlde = @lmnn;
           
           if isunix && exist('lmnn2.m') == 2;
              obj.fhanlde = @lmnn2;
           end
           obj.available = bool;
        end
        
        function s = learnPairwise(obj,X,idxa,idxb,matches)
            if ~obj.available
                s = struct();
                return;
            end
            
            obj.p.knn = 1;
            
            X = X(:,[idxa(matches) idxb(matches)]); %m x d
            y = [1:sum(matches) 1:sum(matches)];%类标号，行人对象的类标号为其ID
            
            tic;
            [s.L, s.Det] = obj.fhanlde(X,consecutiveLabels(y),obj.p.knn, ...
                'maxiter',obj.p.maxiter,'validation',obj.p.validation, ...
                'quiet',obj.p.quiet); 
            s.M = s.L'*s.L;
            s.t = toc;
            s.learnAlgo = obj;
            s.roccolor = obj.p.roccolor;
            obj.learnedParam = s;
        end
        
        function s = learn(obj,X,y)
            if ~obj.available
                s = struct();
                return;
            end
            
            tic;
            [s.L, s.Det] = obj.fhanlde(X,consecutiveLabels(y),obj.p.knn, ...
                'maxiter', obj.p.maxiter,'validation',obj.p.validation, ... 
                'quiet',obj.p.quiet); 
            s.M = s.L'*s.L;
            s.t = toc;
            s.learnAlgo = obj;
            s.roccolor = obj.p.roccolor;
        end
        
        function d = dist(obj, Xa, Xb)
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
            d = sqdist(Xa, Xb,obj.learnedParam.M);
        end
    end    
end

% lmnn2 needs consecutive integers as labels
function ty = consecutiveLabels(y)
    uniqueLabels = unique(y);
    ty = zeros(size(y));
    for cY=1:length(uniqueLabels)
        mask = y == uniqueLabels(cY);
        ty(mask ) = cY;
    end
end