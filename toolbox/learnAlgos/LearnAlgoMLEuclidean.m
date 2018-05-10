%LEARNALGOMLEUCLIDEAN Simulates plain L2 distance as learning algorithm.
classdef LearnAlgoMLEuclidean < LearnAlgo 
    
    properties 
       p %parameters 
       s %struct
       name
       color
       inverseNum
       learnedParam
    end
    
    properties (Constant)
        type = 'identity'
    end
    
    methods
        function obj = LearnAlgoMLEuclidean(p)        
            if nargin < 1
               p = struct(); 
            end
            
            if ~isfield(p,'roccolor')
                p.roccolor = 'r';
            end
            obj.name = 'identity';
           obj.color = 'r';
            obj.p = p;
        end
        
        function s = learnPairwise(obj,X,idxa,idxb,matches)           
            s.M = eye(size(X,1));
            s.t = 0.0;
            s.learnAlgo = obj;
            s.roccolor = obj.p.roccolor;
            obj.learnedParam = s;
        end
        
        function s = learn(obj,X,y)
            s.M = eye(size(X,1));
            s.t = 0.0;
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

