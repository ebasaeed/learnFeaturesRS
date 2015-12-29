%demo for Active set feature generation / selection
%
%
% 2012  Remi FLamary
% 2012  modified by Devis Tuia for
%       remote sensing feature selection (with pre-generation of features).
% 2014  added MLC multiclass selection (D. Tuia)
% 2014  added MLC multiclass and hierarchical selection (D. Tuia)

clear
close all
addpath(genpath('.'))

% method to be used
% - 'SVMl1': Tuia et al, TGRS 2014
% - 'MLCl1l2': Tuia et al, JISPRS (in press)
% - 'MLCl1l2-H': Tuia et al, JISPRS (in press), hierarchical method
method = 'SVMl1';


%Number of experiments
exps = 1 %orginally five

% Number of labeled pixels
pct = 0.75;
% if pct < 1 it is a percentage per class
% if pct is > 1 it is a number of pixels per class


%% Load and prepare image

%Load Data here
noSets = 10;
noMasks = 2; %no of masks per set
noMosaics = 2; %no of mosaics per set
noBands = 10; %no of bands per image
homeDir = '\\KU-ECE-F329-01\Box Sync\ANN Segmentation\ALI [large] data\';


testMaskNo = 1; %default 1
testMosaicNo = 1; %default 1
                  

for setNo= 1:noSets
    for maskNo=1:noMasks
        for mosaicNo=1:noMosaics
            if (maskNo == testMaskNo && mosaicNo == testMosaicNo)
                [inputImg, GT] = openMaskAndImage(homeDir, setNo, maskNo, mosaicNo, noBands);
                GT = double(GT);
                % - data are in immRaw (rows x cols x bands), there are 20 noise bands in
                % it (you may chose to keep them, to show tat they are never selected :)).
                % - ground reference is in GT (rows x cols)

                Sz = size(inputImg);
                SS = Sz;
                
                Y = GT(:);
                YY = find(Y > 0);
                YYY = Y(YY);
                
                ct_sets = [];
                cv_sets = [];
                t_sets = [];
                svmikl = [];
                resikl = [];
                mapikl_sets = zeros(Sz(1)*Sz(2),length(exps));
                
                for run = 1:exps
                    
                    
                    options=struct();
                    
                    
                    options.classifMethod  = method;
                    
                    % set options
                    options.printing = 0; %print weight graphs
                    options.C = 100;
                    options.bandsSelectionMethod = 'manybands';
                    
                    
                    %options.constraintmethod='morpho' ;
                    options.nbitermax = 10;     % nb iter algo, for each OAA subproblem
                    options.nbitergenfeat = 100; % after how many empty Xviol it exits the OAA class
                    options.maxPerBank = 2;     % max number of features selectable for a given bank
                    options.Nscales = 5;        % number of scales / win size to compute on each band
                    options.Nbands = 10;        % How many bands to pick at each generatefeatures
                    options.when2save = 1;     % after how many iter to save the sets
                    
                    % options for the accelerated gradient
                    options_ag.verbose= 0;%: print everything
                    options_ag.log =1;%: log informations
                    options_ag.nbitermax =500 ;%: log informatioouns
                    options_ag.stopvarx=[1e-5];% : threshold for stopping on F changes
                    options_ag.stopvarj=[1e-8];
                    options_ag.nu=1.1;
                    options_ag.L0=1;
                    options_ag.A_0=[];
                    options_ag.C=options.C;
                    
                    optionsFeatSel.tolerance=0.05;
                    optionsFeatSel.nbsamplemax=20;
                    
                    % Options specific for the MLC l1l2 model (Tuia et al., JISPRS in press)
                    options.reg='l1l2'; %regularizer for the MLC model
                    options.lambda = 1e-3;
                    options.tolerance=0.001;
                    
                    %hierarchical mode (add selected features to the batch)
                    % = 0: shallow (like in pcv proceedings) -> AS-Bands
                    % = 1: hierarchical, penalizaiton of deeper features. -> ASH-Bands
                    options.ASH=0;
                    if options.ASH == 1
                        options.gamma = 1.1; % depth penalization parameter
                    end
                    
                    %other things
                    options.randSamp = 0; % to select random filters (for comparison, if needed)
                    options.win = 3; %buffer around the training pixels to avoid contiguous test pixels (pix).
                    options.bandweights = ones(1,Sz(3));%initialize band weights (in case we want to influence probability of selection later)
                    options.currentReal = run;
 
                    %% Learn the features and the classifier
                    
                    % Prepare sets for classif
                    
                    X = reshape(inputImg,Sz(1)*Sz(2),Sz(3));
                    d = size(X,2);
                    
                    
                    %get indices of sets
                    setsr = run-1; %random generator. To have always the same training pts
                    
                    %extract training pixels
                    [~,~,~,~, indices] = ppc(X(YY,1),YYY,pct,setsr);
                    ct = find(indices == 1);
                    %cv = find(indices == 2);
                    
                    
                    %app = training / test = test
                    X2=(X-ones(size(X,1),1)*mean(X))./(ones(size(X,1),1)*std(X));
                    xapp = X2(YY(ct),:);
                    yapp = YYY((ct));
                    %xtest = X2(YY(cv),:);
                    %ytest = YYY((cv));
                    
                    clear X2
                    
                    InfoFeatures.type='original_l1';
                    InfoFeatures.X = X;
                    InfoFeatures.Y = Y;
                    InfoFeatures.YY = YY;
                    InfoFeatures.Sz = Sz;
                    InfoFeatures.ct = ct;
                    %InfoFeatures.cv = cv;
                    InfoFeatures.cl = 0;
                    
                    %Create a mask for test pixels (ym)
                    ym = Y;
                    ym(YY(ct)) = -1;
                    ym = reshape(abs(1-double(ym == -1)),Sz(1),Sz(2));
                    se = strel('square',options.win);
                    ym = imerode(ym,se);
                    yt = max(0,ym(:).*InfoFeatures.Y(:)); %mask for spatially contiguous pixels
                    InfoFeatures.ym = yt;
                    
                    
                    switch options.classifMethod
                        
                        case 'SVMl1'
                            tic
                            [svm,obj,LOG,perIter]=inflinearsvm_manybands(xapp,yapp,options,optionsFeatSel,InfoFeatures);
                            t= toc;
                            %in svm.feat we find the interesting features.
                            
                            
                            [~,map]=inflinearsvmval_manybands(InfoFeatures,svm);
                            figure(setNo), imshow(reshape(map,Sz(1),Sz(2)),[]);
                            
                            % here use your favourite accuracy test code between
                            % - true labels: yt(yt>0)
                            % and
                            % - predictions: map(yt>0)
                            
                            %{
                            res{run}.num = assessment(yt(yt>0),map(yt>0),'class');
                            res{run}.perIter = perIter;
                            res{run}.model = svm;
                            for c = 1:numel(unique(yapp))
                                ActiveFeat{run,c} = perIter{1,c}{1,end}.feat;
                            end
                            disp(['AS,' options.classifMethod ', K, realization ' num2str(run) '=' num2str(res{run}.num.Kappa)])
                           %}
                            
                        case 'MLCl1l2'
                            tic
                            [res{run}]=mc_mlr_as(xapp,yapp,InfoFeatures,options);
                            t = toc;
                            % res{r}{1,iteration}.ActiveFeat contains the active sets at each iteration (the last has the final active set)
                            
                            ActiveFeat = res{run}{end}.ActiveFeat;
                            disp(['AS,' options.classifMethod ', K, realization ' num2str(run) '=' num2str(res{run}{end}.num.Kappa)])
                            
                        case 'MLCl1l2-H'
                            tic
                            [res{run},InfoFeatures,ActiveFeat{run}]=mc_mlr_as_deep3(xapp,yapp,InfoFeatures,options);
                            t = toc;
                            % res{r}{1,iteration}.ActiveFeat contains the active sets at each iteration (the last has the final active set)
                            % ActiveFeat{r} contains the final active set)
                            
                            disp(['AS,' options.classifMethod ', K, realization ' num2str(run) '=' num2str(res{run}{end}.num.Kappa)])
                     
                    end
                    
                    % Validation after selection
                    ct_sets = [ct_sets ct];
                    %cv_sets = [cv_sets cv];
                    t_sets = [t_sets t];
                    
                    
                    %save(['./res' options.classifMethod '_' num2str(pct) 'perclass.mat'],'res','ct_sets','cv_sets','t_sets','options','ActiveFeat')
                end
                
            end
        end
    end
end







