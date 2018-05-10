%LEARNALGOITML Wrapper class to the actual ITML code
classdef LearnAlgoITML < LearnAlgo
    
    properties 
       p %parameters 
       s %struct
       available
       name
       color
       inverseNum
       learnedParam
    end
    
    properties (Constant)
        type = 'itml'
    end
    
    methods
        function obj = LearnAlgoITML(p)
           if nargin < 1
              p = struct(); 
           end
           
           if ~isfield(p,'roccolor')
                p.roccolor = 'b';
           end
           obj.name = 'itml';
           obj.color = 'b';
           obj.p  = p;
           check(obj);
        end
        
        function bool = check(obj)
           bool = exist('ItmlAlg') ~= 0;
           if ~bool
               fprintf('Sorry %s not available\n',obj.type);
           end
           obj.available = bool;
        end
        
        function s = learnPairwise(obj,X,idxa,idxb,matches)
            if ~obj.available
                s = struct();
                return;
            end
            
            tic;
            s.M = PairMetricLearning(@ItmlAlg, idxa', idxb', matches, X');
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
            s.M = MetricLearning(@ItmlAlg, y', X');
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

