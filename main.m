%This is the main script for the code needed to create fuzzy models for my thesis
%Done by: André Miranda n84205 @ Instituto superior Técnico
%            andre.lima.miranda@tecnico.ulisboa.pt

%Loading the required variables
clear
clc
close
%[notall,header]=xlsread("all_compli.xls");    %This data is not normalized
%[notall,header]=xlsread("all_death.xls");       %This data is not normalized

%[notall,header]=xlsread("all_compli_norm.xls");    %This data is normalized
[notall,header]=xlsread("all_death_norm.xls");       %This data is normalized


How_many_features=15;   %This variable chooses the number of features used
algorithm=0;            %This number may be 0-Fuzzy c-means or
                                            %1-Subtractive clustering
sample=0;               %This variable controls oversampling - 1 or undersampling - 0



rng('default')  %Sets the random key to 'default', making the results repeatable

for counter=2:10

for feature_selection=6:6
    if header{end}(1)=='c'
        
        %Complications
        if feature_selection==1
            %# 1 for random set of features
            all=notall(:,[randi(51,[1,How_many_features]) length(notall(1,:))]);
            
        elseif feature_selection==2
            %# 2 for Pearson
            strings=["PP contaminação peritoneal","tipo cirurgia","ACS estado funcional","motivo admissão UCI","ACS sépsis sistémica","ASA","proveniência","PP nº procedimentos","ARISCAT anemia pré-operativa","ACS dependente ventilador","ARISCAT SpO2 ","ACS ascite","PP hemoglobina","PP respiratório","ACS dispneia"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==3
            %# 3 for Spearman
            strings=["tipo cirurgia", "PP contaminação peritoneal", "ACS estado funcional","ACS sépsis sistémica", "motivo admissão UCI", "proveniência", "ASA","ARISCAT anemia pré-operativa", "PP nº procedimentos","ACS dependente ventilador", "ARISCAT SpO2 ", "PP respiratório","ACS ascite", "ACS dispneia", "PP hemoglobina"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==4
            %# 4 for chi2
            strings=["proveniência","motivo admissão UCI","tipo cirurgia","PP respiratório","PP hemoglobina","PP nº procedimentos","PP contaminação peritoneal","ACS estado funcional","ACS ascite","ACS sépsis sistémica","ACS dependente ventilador","ACS dispneia","ARISCAT SpO2 ","ARISCAT anemia pré-operativa","CHARLSON Doença Renal Crónica Moderada a Severa"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==5
            %# 5 for Lasso
            strings=["género", "motivo admissão UCI", "tipo cirurgia", "PP respiratório","PP perda sangue", "PP contaminação peritoneal", "ACS estado funcional","ARISCAT anemia pré-operativa", "ARISCAT duração cirurgia","CHARLSON Doença Renal Crónica Moderada a Severa"];
            
            if How_many_features>length(strings)
                lasso_number_features=length(strings);
            else
                lasso_number_features=How_many_features;
            end
            
            all=notall(:,[find_feature_index(header,strings(1:lasso_number_features)) length(notall(1,:))]);
            
        elseif feature_selection==6
            %# 6 for Mutual information
            strings=["motivo admissão UCI", "tipo cirurgia", "ASA", "PP cardíaco","PP respiratório", "PP hemoglobina", "PP leucócitos", "PP ureia","PP potássio", "PP contaminação peritoneal", "ACS estado funcional","ACS dependente ventilador", "ACS dispneia","ACS insuficiência renal aguda", "ACS peso"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==7
            %# 7 for a weighted (all methods) method
            strings=["tipo cirurgia", "motivo admissão UCI", "PP contaminação peritoneal","ACS estado funcional", "proveniência", "ACS sépsis sistémica","PP respiratório", "PP nº procedimentos","ARISCAT anemia pré-operativa", "ASA", "ACS dependente ventilador","PP hemoglobina", "ACS ascite", "ARISCAT SpO2 ", "género"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        end
        
        
        if algorithm==0
            %Fuzzy c means clustering
            %Number_of_clusters=5;
            Number_of_clusters=counter;
            opt_fcm = genfisOptions('FCMClustering');
            opt_fcm.NumClusters = Number_of_clusters;
            opt_fcm.Exponent = 1.1;             %Manages the overlap between clusters, the higher the more overlap
            opt_fcm.MaxNumIteration = 1000;       %Self explanatory
            opt_fcm.MinImprovement = 0.1;    %Stopping criteria
            opt_fcm.Verbose = 0;                %Shows method
            
        elseif algorithm==1
            %Subtractive custering
            
            options = genfisOptions('SubtractiveClustering');
            options.ClusterInfluenceRange =[1];      %see https://www.mathworks.com/help/fuzzy/subclust.html#bvmwqj7-clusterInfluenceRange
            options.DataScale = 'auto';
            options.SquashFactor = 1.5;        %Determines the ability of outliers to generate more clusters the higher it goes
            options.AcceptRatio = 0.8;       %Acceptance to a cluster value [0-1] needs to be larger than rejection
            options.RejectRatio = 0.5;       %Rejection to a cluster value [0-1]
            options.Verbose = 0;           %Shows function outputs yes 1 or no 0
        end
        
    elseif header{end}(1)=="ó"
        
        %Death
        if feature_selection==1
            %# 1 for random set of features
            all=notall(:,[randi(51,[1,15]) length(notall(1,:))]);
            
        elseif feature_selection==2
            %# 2 for Pearson
            strings=["ACS estado funcional","ASA","PP hemoglobina","PP estado da malignidade","ACS cancro disseminado","PP contaminação peritoneal","especialidade_COD","ACS peso","ACS ICC","ACS sépsis sistémica","PP pulsação arterial","ACS ascite","ARISCAT anemia pré-operativa","PP leucócitos","PP respiratório"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==3
            %# 3 for Spearman
            strings=["ACS estado funcional", "ASA", "PP contaminação peritoneal","PP estado da malignidade", "PP hemoglobina", "ACS cancro disseminado","especialidade_COD", "ACS peso", "ACS ICC", "ACS sépsis sistémica","ACS ascite", "ARISCAT anemia pré-operativa", "PP leucócitos","PP pulsação arterial", "ARISCAT SpO2 "];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==4
            %# 4 for chi2
            strings=["tipo cirurgia","PP respiratório","PP hemoglobina","PP leucócitos","PP contaminação peritoneal","ACS estado funcional","ACS esteróides","ACS ascite","ACS sépsis sistémica","ACS cancro disseminado","ACS ICC","ACS peso","ARISCAT anemia pré-operativa","CHARLSON Doença Renal Crónica Moderada a Severa","CHARLSON AVC ou Ataque Isquémico Transitório"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==5
            %# 5 for Lasso
            strings=["género", "especialidade_COD", "PP hemoglobina", "PP perda sangue","ACS estado funcional", "ACS cancro disseminado", "ACS ICC","ACS peso"];
            
            if How_many_features>length(strings)
                lasso_number_features=length(strings);
            else
                lasso_number_features=How_many_features;
            end
            
            all=notall(:,[find_feature_index(header,strings(1:lasso_number_features)) length(notall(1,:))]);
            
        elseif feature_selection==6
            %# 6 for Mutual information
            strings=["tipo cirurgia", "especialidade_COD", "PP cardíaco","PP pulsação arterial", "PP hemoglobina", "PP leucócitos","PP contaminação peritoneal", "PP estado da malignidade","ACS estado funcional", "ACS esteróides", "ACS DPOC", "ACS altura","ACS peso", "ARISCAT SpO2 ", "ARISCAT anemia pré-operativa"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        elseif feature_selection==7
            %# 7 for a weighted (all methods) method
            strings=["ACS estado funcional", "PP hemoglobina", "PP contaminação peritoneal","ACS cancro disseminado", "ASA", "especialidade_COD","PP estado da malignidade", "ACS ICC", "ACS peso","ACS sépsis sistémica", "ACS ascite", "PP leucócitos","PP respiratório", "tipo cirurgia", "ARISCAT anemia pré-operativa"];
            all=notall(:,[find_feature_index(header,strings(1:How_many_features)) length(notall(1,:))]);
            
        end
        
        if algorithm==0
            %Fuzzy c means clustering
            %Number_of_clusters=5;
            Number_of_clusters=counter;
            opt_fcm = genfisOptions('FCMClustering');
            opt_fcm.NumClusters = Number_of_clusters;
            opt_fcm.Exponent = 1.1;             %Manages the overlap between clusters, the higher the more overlap
            opt_fcm.MaxNumIteration = 1000;       %Self explanatory
            opt_fcm.MinImprovement = 0.1;    %Stopping criteria
            opt_fcm.Verbose = 0;                %Shows method
            
        elseif algorithm==1
            %Subtractive custering
            
            options = genfisOptions('SubtractiveClustering');
            options.ClusterInfluenceRange =[0.8];      %see https://www.mathworks.com/help/fuzzy/subclust.html#bvmwqj7-clusterInfluenceRange
            options.DataScale = 'auto';
            options.SquashFactor = 1.25;        %Determines the ability of outliers to generate more clusters the higher it goes
            options.AcceptRatio = 0.6;       %Acceptance to a cluster value [0-1] needs to be larger than rejection
            options.RejectRatio = 0.3;       %Rejection to a cluster value [0-1]
            options.Verbose = 0;           %Shows function outputs yes 1 or no 0
        end
        
    end
    
    
    %     for j=1:1
    %
    %
    %         data_index=cvpartition(all(:,end),"Kfold",5);
    %
    %         for i=1:data_index.NumTestSets                      %Cross validation
    %
    %             x_train=all(data_index.training(i),1:end-1);
    %             x_test=all(data_index.test(i),1:end-1);
    %
    %             y_train=all(data_index.training(i),end);
    %             y_test=all(data_index.test(i),end);
    %
    %             Sis_fcm(i) = genfis(x_train,y_train,opt_fcm);
    %
    %             data_index_val=cvpartition(y_train,"holdout",0.1);
    %
    %             x_val=x_train(data_index_val.test,:);         %Creating the validation data vectors
    %             y_val=y_train(data_index_val.test,:);
    %
    %             x_train=x_train(data_index_val.training,:);       %Training data
    %             y_train=y_train(data_index_val.training,:);
    %
    %
    %             opt = anfisOptions("InitialFIS",Sis_fcm(i));        %Coding parameters of the anfis function
    %             opt.ValidationData = [x_val y_val];
    %             opt.EpochNumber = 200;
    %             opt.InitialStepSize= 1;
    %
    %
    %
    %
    %             [~,trainError,stepSize,Trained_model(i),chkError] = anfis([x_train y_train],opt);
    %
    %             y_test_predicted_auc=evalfis(x_test,Trained_model(i));
    %
    %
    %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %             %Highlight that the prediction is roundes to look like a
    %             %classification task
    %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %             for k=1:length(y_test_predicted_auc)        %Goes through the output vector to approxiate outliers into the supposed interval more than one equals unity, less than zero equals nullity
    %                 if y_test_predicted_auc(k)>1
    %                     y_test_predicted_auc(k)=1;
    %                 elseif  y_test_predicted_auc(k)<0
    %                     y_test_predicted_auc(k)=0;
    %                 end
    %             end
    %
    %
    %             y_test_predicted=round_them(y_test_predicted_auc,0.5);      %The rounding can have an  adjustable threshold, ths hould be investigated
    %
    %
    %             [scores_cross_val(i),conf(i),~]=check_scores(y_test,y_test_predicted,y_test_predicted_auc);
    %
    % %             figure('Name','Training and testing error evolution','NumberTitle','off')
    % %             plot([trainError chkError])
    % %             title(strcat("Model ",num2str(i)," training vs testing error for ",num2str(Number_of_clusters)," clusters"))
    % %             legend("Training Error","Testing error","Location","NorthWest")
    % %             pause(5)
    %
    %         end
    %         scores(j).acc=mean([scores_cross_val.acc]);
    %         scores(j).f1=mean([scores_cross_val.f1]);
    %         scores(j).mcc=mean([scores_cross_val.mcc]);
    %         scores(j).auc=mean([scores_cross_val.auc]);
    %         scores(j).kappa=mean([scores_cross_val.kappa]);
    %         scores(j).recall=mean([scores_cross_val.recall]);
    %         conf(j).TP=mean([conf.TP]);
    %         conf(j).FN=mean([conf.FN]);
    %         conf(j).FP=mean([conf.FP]);
    %         conf(j).TN=mean([conf.TN]);
    %         figure('Name','Training and testing error evolution','NumberTitle','off')
    %         plot([trainError chkError])
    %         title(strcat("Model ",num2str(j)," training vs testing error"))
    %         legend("Training Error","Testing error","Location","NorthWest")
    %
    %     end
    
    %trial_number=1;
    for number_features=How_many_features
        
        if feature_selection==5
            if How_many_features>length(strings)
                lasso_number_features=length(strings);
                all=all(:,[1:lasso_number_features length(all(1,:))]);
            else
                all=all(:,[1:number_features length(all(1,:))]);
            end
        else
        all=all(:,[1:number_features length(all(1,:))]);    
        end
        
       
        
        for j=1:100
            
            
            data_index=cvpartition(all(:,end),"holdout",0.25);
            
            
            x_train=all(data_index.training,1:end-1);
            x_test=all(data_index.test,1:end-1);
            
            y_train=all(data_index.training,end);
            y_test=all(data_index.test,end);
            
            if sample==1
                [x_train,y_train]=oversample(x_train,y_train);
            elseif sample==0
                [x_train,y_train]=undersample(x_train,y_train);
            end
            
            Sis_fcm(j) = genfis(x_train,y_train,opt_fcm);
            
            data_index_val=cvpartition(y_train,"holdout",0.1);
            
            x_val=x_train(data_index_val.test,:);         %Creating the validation data vectors
            y_val=y_train(data_index_val.test,:);
            
            x_train=x_train(data_index_val.training,:);       %Training data
            y_train=y_train(data_index_val.training,:);
            
            
            opt = anfisOptions("InitialFIS",Sis_fcm(j));        %Coding parameters of the anfis function
            opt.ValidationData = [x_val y_val];
            opt.EpochNumber = 100;
            opt.InitialStepSize= 1;
             opt.DisplayErrorValues=false;
             opt.DisplayStepSize=false;
             opt.DisplayANFISInformation=false;
%             opt.DisplayFinalResults=false;
            
            
            [~,trainError,stepSize,Trained_model(j),chkError] = anfis([x_train y_train],opt);
            
            y_test_predicted_auc=evalfis(Trained_model(j),x_test);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Highlight that the prediction is roundes to look like a
            %classification task
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            for k=1:length(y_test_predicted_auc)        %Goes through the output vector to approxiate outliers into the supposed interval more than one equals unity, less than zero equals nullity
                if y_test_predicted_auc(k)>1
                    y_test_predicted_auc(k)=1;
                elseif  y_test_predicted_auc(k)<0
                    y_test_predicted_auc(k)=0;
                end
            end
            
            y_test_predicted=round_them(y_test_predicted_auc,0.5);      %The rounding can have an  adjustable threshold, ths hould be investigated
            
            
            [scores_cross_val,conf,~]=check_scores(y_test,y_test_predicted,y_test_predicted_auc);
            
            
            scores(j).acc=mean([scores_cross_val.acc]);
            scores(j).f1=mean([scores_cross_val.f1]);
            scores(j).mcc=mean([scores_cross_val.mcc]);
            scores(j).auc=mean([scores_cross_val.auc]);
            scores(j).kappa=mean([scores_cross_val.kappa]);
            scores(j).recall=mean([scores_cross_val.recall]);
            conf(j).TP=mean([conf.TP]);
            conf(j).FN=mean([conf.FN]);
            conf(j).FP=mean([conf.FP]);
            conf(j).TN=mean([conf.TN]);
            
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Subtractive custering
        
        % options = genfisOptions('SubtractiveClustering');
        % options.ClusterInfluenceRange =[0.8];      %see https://www.mathworks.com/help/fuzzy/subclust.html#bvmwqj7-clusterInfluenceRange
        % options.DataScale = 'auto';
        % options.SquashFactor = 1.25;        %Determines the ability of outliers to generate more clusters the higher it goes
        % options.AcceptRatio = 0.6;       %Acceptance to a cluster value [0-1] needs to be larger than rejection
        % options.RejectRatio = 0.3;       %Rejection to a cluster value [0-1]
        % options.Verbose = 1;           %Shows function outputs yes 1 or no 0
        %
        %
        %
        % for j=1:1
        %
        %
        %     data_index=cvpartition(all(:,end),"Kfold",5);
        %
        %     for i=1:data_index.NumTestSets                      %Cross validation
        %
        %         x_train=all(data_index.training(i),1:end-1);
        %         x_test=all(data_index.test(i),1:end-1);
        %
        %         y_train=all(data_index.training(i),end);
        %         y_test=all(data_index.test(i),end);
        %
        %         Sis_fcm(i) = genfis(x_train,y_train,options);
        %
        %         data_index_val=cvpartition(y_train,"holdout",0.1);
        %
        %         x_val=x_train(data_index_val.test,:);         %Creating the validation data vectors
        %         y_val=y_train(data_index_val.test,:);
        %
        %         x_train=x_train(data_index_val.training,:);       %Training data
        %         y_train=y_train(data_index_val.training,:);
        %
        %
        %         opt = anfisOptions("InitialFIS",Sis_fcm(i));        %Coding parameters of the anfis function
        %         opt.ValidationData = [x_val y_val];
        %         opt.EpochNumber = 100;
        %         opt.InitialStepSize= 1;
        %
        %         [~,trainError,stepSize,Trained_model(i),chkError] = anfis([x_train y_train],opt);
        %
        %         y_test_predicted_auc=evalfis(x_test,Trained_model(i));
        %
        %
        %         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         %%Highlight that the prediction is rounded to look like a
        %         %%classification task
        %         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %         y_test_predicted=round(y_test_predicted_auc);      %The rounding can have an  adjustable threshold, ths hould be investigated
        %
        %
        %         for k=1:length(y_test_predicted)        %Goes through the output vector to approxiate outliers into the supposed interval more than one equals unity, less than zero equals nullity
        %             if y_test_predicted(k)>1
        %                 y_test_predicted(k)=1;
        %             elseif  y_test_predicted(k)<0
        %                 y_test_predicted(k)=0;
        %             end
        %         end
        %
        %         [scores_cross_val(i),conf(i),~]=check_scores(y_test,y_test_predicted,y_test_predicted_auc);
        %
        %         figure('Name','Training and testing error evolution','NumberTitle','off')
        %         plot([trainError chkError])
        %         title(strcat("Model ",num2str(i)," training vs testing error for Subtractive clustering"))
        %         legend("Training Error","Testing error","Location","NorthWest")
        %         pause(5)
        %
        %     end
        %
        %
        %     scores(j).acc=mean([scores_cross_val.acc]);
        %     scores(j).f1=mean([scores_cross_val.f1]);
        %     scores(j).mcc=mean([scores_cross_val.mcc]);
        %     scores(j).auc=mean([scores_cross_val.auc]);
        %     scores(j).kappa=mean([scores_cross_val.kappa]);
        %     scores(j).recall=mean([scores_cross_val.recall]);
        %     conf(j).TP=mean([conf.TP]);
        %     conf(j).FN=mean([conf.FN]);
        %     conf(j).FP=mean([conf.FP]);
        %     conf(j).TN=mean([conf.TN]);
        %
        % %     figure('Name','Training and testing error evolution','NumberTitle','off')
        % %     plot([trainError chkError])
        % %     title(strcat("Model ",num2str(j)," training vs testing error"))
        % %     legend("Training Error","Testing error","Location","NorthWest")
        % end
        
        
        
        %Plot the labels vs the predicted labels
        
        
        % figure
        % plot(linspace(1,length(x_train),length(x_train)),y_train,linspace(1,length(x_train),length(x_train)),evalfis(x_train,Trained_model));
        
        % figure('Name','Test data spline','NumberTitle','off')
        % plot(linspace(1,length(x_test),length(x_test)),y_test,linspace(1,length(x_test),length(x_test)),y_test_predicted);
        % title(strcat("Test data in ANFIS model with FCM clustering with ",num2str(Number_of_clusters)," clusters"))
        % ylim([-0.5 1.5])
        % legend("Training Data","ANFIS Output","Location","NorthWest")
        %
        % x_index=1;for i=2:length(y_test),x_index(i)=x_index(i-1)+1;end %Creates an index vecor to plot the data
        %
        % figure('Name','Test data scatter','NumberTitle','off')
        % plot(x_index,y_test,".b",x_index,y_test_predicted,"or");        %Scatter plot of the target vs predicted class vectors
        % title(strcat("Test data in ANFIS model with FCM clustering with ",num2str(Number_of_clusters)," clusters"))
        % ylim([-0.5 1.5])
        % legend("Training Data","ANFIS Output","Location","NorthWest")
        
        
        
        
        %display(scores)
        
        [acc,f1,mcc,auc,kappa,recall]=score_splitter(scores);   %Creates 6 arrays and 1 struct with the scores separated
        
        %Finding the best metrics and respective arguments
        
        indices_to_keep=~isnan(f1);
        
        acc=acc(indices_to_keep);
        f1=f1(indices_to_keep);
        mcc=mcc(indices_to_keep);
        auc=auc(indices_to_keep);
        kappa=kappa(indices_to_keep);
        recall=recall(indices_to_keep);
        
        
        [~,bestie]=max(recall);
        
        %trial_number=feature_selection;
        trial_number=counter;
        
        mean_std(2*trial_number-1,:)=[mean(acc) mean(f1) mean(mcc) mean(auc) mean(kappa) mean(recall)]; %keeps the mean and standard deviation among models
        mean_std(2*trial_number,:)=[std(acc) std(f1) std(mcc) std(auc) std(kappa) std(recall)];
        
        
        best_of_all(trial_number,:)=[acc(bestie) f1(bestie) mcc(bestie) auc(bestie) kappa(bestie) recall(bestie)]; %%%%keeps the best models scores
        
        %trial_number=trial_number+1;
        
    end
    
    
    
%     fprintf('Accuracy --> %8.3f (std:%6.3f)\n',mean(acc),std([scores_cross_val.acc]));
%     fprintf('F1_score --> %8.3f (std:%6.3f)\n',mean(f1),std([scores_cross_val.f1]));
%     fprintf('Mathews  --> %8.3f (std:%6.3f)\n',mean(mcc),std([scores_cross_val.mcc]));
%     fprintf('ROC AUC  --> %8.3f (std:%6.3f)\n',mean(auc),std([scores_cross_val.auc]));
%     fprintf('Kappa    --> %8.3f (std:%6.3f)\n',mean(kappa),std([scores_cross_val.kappa]));
%     fprintf('Recall   --> %8.3f (std:%6.3f)\n',mean(recall),std([scores_cross_val.recall]));
    
end



end

% fprintf('\n');
% fprintf('\n');
% fprintf('Avg metrics across %d models: \n',length(scores_cross_val));
% fprintf('Accuracy --> %8.3f (std:%6.3f)\n',mean(acc),std([scores_cross_val.acc]));
% fprintf('F1_score --> %8.3f (std:%6.3f)\n',mean(f1),std([scores_cross_val.f1]));
% fprintf('Mathews  --> %8.3f (std:%6.3f)\n',mean(mcc),std([scores_cross_val.mcc]));
% fprintf('ROC AUC  --> %8.3f (std:%6.3f)\n',mean(auc),std([scores_cross_val.auc]));
% fprintf('Kappa    --> %8.3f (std:%6.3f)\n',mean(kappa),std([scores_cross_val.kappa]));
% fprintf('Recall   --> %8.3f (std:%6.3f)\n',mean(recall),std([scores_cross_val.recall]));
% fprintf('ConfTP   --> %8.3f (std:%3.f)\n',mean([conf.TP]),std([conf.TP]));
% fprintf('ConfFN   --> %8.3f (std:%3.f)\n',mean([conf.FN]),std([conf.FN]));
% fprintf('ConfFP   --> %8.3f (std:%3.f)\n',mean([conf.FP]),std([conf.FP]));
% fprintf('ConfTN   --> %8.3f (std:%3.f)\n',mean([conf.TN]),std([conf.TN]));

% fprintf('\n');
% fprintf('\n');
% fprintf('Avg metrics across %d models: \n',length(acc));
% fprintf('Accuracy --> %8.3f (std:%6.3f)\n',mean(acc),std(acc));
% fprintf('F1_score --> %8.3f (std:%6.3f)\n',mean(f1),std(f1));
% fprintf('Mathews  --> %8.3f (std:%6.3f)\n',mean(mcc),std(mcc));
% fprintf('ROC AUC  --> %8.3f (std:%6.3f)\n',mean(auc),std(auc));
% fprintf('Kappa    --> %8.3f (std:%6.3f)\n',mean(kappa),std(kappa));
% fprintf('Recall   --> %8.3f (std:%6.3f)\n',mean(recall),std(recall));
% fprintf('ConfTP   --> %8.3f (std:%3.f)\n',mean([conf.TP]),std([conf.TP]));
% fprintf('ConfFN   --> %8.3f (std:%3.f)\n',mean([conf.FN]),std([conf.FN]));
% fprintf('ConfFP   --> %8.3f (std:%3.f)\n',mean([conf.FP]),std([conf.FP]));
% fprintf('ConfTN   --> %8.3f (std:%3.f)\n',mean([conf.TN]),std([conf.TN]));
% 

% fprintf('\n');
% fprintf('\n');
% fprintf('Avg metrics across %d models: \n',length(acc));
% fprintf('Accuracy --> %8.3f (index:%3.f)\n',mean_acc,i_acc);
% fprintf('F1_score --> %8.3f (index:%3.f)\n',mean_f1,i_f1);
% fprintf('Mathews  --> %8.3f (index:%3.f)\n',mean_mcc,i_mcc);
% fprintf('ROC AUC  --> %8.3f (index:%3.f)\n',mean_auc,i_auc);
% fprintf('Kappa    --> %8.3f (index:%3.f)\n',mean_kappa,i_kappa);
% fprintf('Recall   --> %8.3f (index:%3.f)\n',mean_recall,i_recall);




