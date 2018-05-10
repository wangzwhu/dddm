tic;
clc; clear all; close all;
addpath(genpath('./SLEP_package_4.1/SLEP'));
DATA_OUT_DIR = fullfile('dataOut');
run('./toolbox/init.m');

TIMES = 1;
data_type = 1;      %  1-viper,  2-cuhk, 3-prid
method_type = 1;    %  1-kissme, 2-lmnn, 3-mal

%% Set up parameters  &  Load Features
if data_type == 1
    params.numCoeffs = 40; %dimensionality reduction by PCA to 34 dimension  /    VIPER 40,  CUHK  361
    params.N = 632;        %number of image pairs, 316 to train 316 to test  /    VIPER 632, CUHK  971 
    params.numFolds = 1;   %number of random train/test splits
    params.saveDir = fullfile(DATA_OUT_DIR,'all');
    params.pmetric = 0;
    load(fullfile(DATA_OUT_DIR,'viper_features.mat')); 
%   load(fullfile(DATA_OUT_DIR,'features_VIPeR.mat'));
%     ux = features;
%     idxa = 1:632;
%     idxb = 633:1264;
elseif data_type == 2
    params.numCoeffs = 40;  %dimensionality reduction by PCA to 34 dimension  /    VIPER 40,  CUHK  50
    params.N = 970;         %number of image pairs, 316 to train 316 to test  /  CUHK  971 
    params.numFolds = 1;    %number of random train/test splits
    params.saveDir = fullfile(DATA_OUT_DIR,'all');
    params.pmetric = 0;
    load(fullfile(DATA_OUT_DIR,'cuhk-fea-12-16-PCA400.mat'));
    idxa = idxa(:,1:end-1);
    idxb = idxb - 1;
    idxb = idxb(:,1:end-1);
    ux_PCA1 = ux_PCA(:,1:970);
    ux_PCA2 = ux_PCA(:,972:1941);
    ux_PCA = [ux_PCA1 ux_PCA2];
    [ux_PCA, PC, V] = pca1(ux_PCA);
elseif data_type == 3
    params.numCoeffs = 16;
    params.N = 200;
    params.numFolds = 1;
    params.saveDir = fullfile(DATA_OUT_DIR,'all');
    params.pmetric = 0;
    load(fullfile(DATA_OUT_DIR,'PRID_a_b_Feature_400.mat'));   
    idxb = idxb(:,1:200);
    ux = ux(:,1:400);
    [ux, PC, V] = pca1(ux);
end



%% Cross-validate over a number of runs
cmc =[];
cmc2=[];
cmc3=[];
cmc4=[];

for k=1:TIMES
    if method_type == 1
        pair_metric_learn_algs = {...
            LearnAlgoKISSME(params), ...
        };
    elseif method_type == 2
        pair_metric_learn_algs = {...
           LearnAlgoLMNN(), ...
        };  
    elseif method_type == 3
        pair_metric_learn_algs = {...
        LearnAlgoMahal()...
        };
    end
    clear ds;
    if data_type == 1
        [ ds ] = CrossValidateViper(struct(), pair_metric_learn_algs,ux(1:params.numCoeffs,:),idxa,idxb,params);
    elseif data_type == 2
        [ ds ] = CrossValidateViper(struct(), pair_metric_learn_algs,ux_PCA(1:params.numCoeffs,:),idxa,idxb,params);
    elseif data_type == 3
        [ ds ] = CrossValidateViper(struct(), pair_metric_learn_algs,ux(1:params.numCoeffs,:),idxa,idxb,params);
    end
    names = fieldnames(ds);
    for nameCounter=1:length(names)
        s = [ds.(names{nameCounter})];
        ms.(names{nameCounter}).cmc = cat(1,s.cmc)./(params.N/2);
        ms.(names{nameCounter}).cmc2 = cat(1,s.cmc2)./(params.N/2);
        ms.(names{nameCounter}).cmc3 = cat(1,s.cmc3)./(params.N/2);
        ms.(names{nameCounter}).cmc4 = cat(1,s.cmc4)./(params.N/2);
        ms.(names{nameCounter}).roccolor = s.roccolor;
    end
    if k==1
        cmc  = ms.(names{nameCounter}).cmc;
        cmc2 = ms.(names{nameCounter}).cmc2;
        cmc3 = ms.(names{nameCounter}).cmc3;
        cmc4 = ms.(names{nameCounter}).cmc4;
    else
        cmc  = cmc + ms.(names{nameCounter}).cmc;
        cmc2 = cmc2 + ms.(names{nameCounter}).cmc2;
        cmc3 = cmc3 + ms.(names{nameCounter}).cmc3;
        cmc4 = cmc4 + ms.(names{nameCounter}).cmc4;  
    end
   
end

%% Plot Cumulative Matching Characteristic (CMC) Curves
h = figure;
names = fieldnames(ms);
for nameCounter=1:length(names)
   hold on; 
   plot(cmc/TIMES,'Color','g','LineWidth',2);
   plot(cmc2/TIMES,'Color','b','LineWidth',2);
   plot(cmc3/TIMES,'Color','k','LineWidth',2);
   plot(cmc4/TIMES,'Color','r','LineWidth',2);
end
 
if data_type == 1
    title('cmc VIPeR');
elseif data_type == 2
    title('cmc CUHK');
elseif data_type == 3
    title('cmc PRID');
end

box('on');
set(gca,'XTick',[0 1 5 10 20 30 40 50 60 80 100]);

ylabel('Matches');
xlabel('Rank');
xlim([0 100]);
ylim([0 1]);
hold off;
grid on;

if method_type == 1
    show_method = 'KISSME';
elseif method_type == 2
    show_method = 'LMNN';
elseif method_type == 2
    show_method = 'MAL';
end
legend(show_method,'Cross-view Support Consistency','Cross-view Projection Consistency','Both',4); 



