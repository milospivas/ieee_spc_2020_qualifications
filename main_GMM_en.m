%% IEEE Signal Processing Cup 2020 - Qualifications
% % Team 2 : GMM
%% Report
% Defining the problem
% 5 normal and 5 abnormal recordings are given as ROS .bag files.
% 
% The task is to implement models which will give a prediction wheter each sample 
% in the recording is normal/abnormal based on measurements from .bag files.
% Visualization:
% Images were extracted from "/pylon_camera_node/image_raw" topic, rotated 180 
% degrees and saved as video files with 4fps framerate.
% 
% From extracted videos we see that the recordings come from some UAV, where 
% the normal recordings were made during stable flight without fast movement while 
% abnormal recordings were made during very unstable flight with sudden movements.
% Preanalysis
% Every recording has around hundred mesurements but we don't have class labels.
% 
% => We can't use classic classification.
% 
% => Possible solution: Gaussian Mixture Model
% Measurement analysis
% Based on extracted videos, the following measurements were chosen as relevant 
% for analysis:
% 
% "/mavros/global_position/local"
% 
% "/mavros/imu/data"
% 
% "/mavros/imu/mag"
% 
% "/mavros/global_position/compass_hdg"
% 
% Time plots and histograms show that the oscillations of some measurements 
% are much larger on abnormal recordings than normal ones.
% 
% That means that the derivatives of those signals reach higher absolute values 
% on abnormal recordings, which means that variances of those measurements are 
% larger than on normal recordings so they are deemed as more informative.
% 
% Physical quantities that satisfy the above description are orientation, linear 
% velocity, magnetic field and compass heading.
% 
% Compass heading was discarded because of correlation with magnetic field.
% 
% Their derivatives are angular velocity, linear acceleration (already available 
% from IMU), and derivative of magnetic field.
% 
% All values are 3D, so the feature vector is 9D.
% 
% Feature engineering
% We added a "lookBack" - adds k last samples to each sample, allowing the ML 
% algorithm to "look back" in time.
% 
% Gaussian Mixture Model
% Implemented GMM with 2 classes.
% 
% A sample is classified as abnormal if its posterior probability of abnormal 
% GMM component is larger from the normal one.
% 
% Used regularization, and selected the best regularization parameter on CV 
% set, via hold-out cross-validation.
% 
% Models are evaluated via silhouette analysis.
% 
% Silhouette analysis
% Since the GMM components are overlapping and concentric, evaluating silhouettes 
% on raw data would produce nonsensical results.
% 
% The data is first mapped to a distance metric:
% 
% Mahalanobis distance from the normal GMM components minus the distance from 
% abnromal component.
% 
% That creates a nice separation between normal and abnormal samples.
% 
% % Overview of files:

%     addDerivative      - calculates derivatives of selected columns of the table and adds the derivative columns.
%     extractImages      - Extracts images from a bag object
%     files2bag          - Loads all .bag files from given directory into a cell array of bag objects.
%     gmm2dVisualisation - 2D visualization of a GMM. Draws a contour plot and a surf plot, for every GMM component.
%     gmmFit             - Fits GMM to data. Computes Gaussian mixture model on data Z with given
%                           number of classes and regularization value Lambda.
%     lookBack           - For each row in the input matrix, append k last rows to the right.
%     mapFrames          - Matches IMU measurements with corresponding frames.
%                           Sorts IMU timestamps and measurements into table assigning each measurment
%                           set a serial number of the frame that first comes after
%     silhouetteData     - Calculates data to be used for silhouette analysis
%                           Maps the data points using Mahalanobis distance.
%     silhouetteEval     - Evaluate the silhouettes for choosing best model.
%     silhouetteVisualization - Visualize silhouette data and draw silhouettes.
%                           Draws a histogram of input data for silhouette analysis.
%                           Draws the silhouettes.
%     splitData          - Split the given data into training, CV and test sets using given percent values for split.

%     tables.mat         - saved data, extracted from the bag files
%     GMM.mat            - saved trained model with parameters for data preprocessing

%     main_GMM_en            - Main program.
%% 
% %% Implementation Work directories for normal and abnormal data.

clear;

% change if needed
workDirNormal = '03_normal';
workDirAbnormal = '04_abnormal';
%% Loading the data
% Loading .bag files, selecting relevant measurements and converting into tables.

bagsNormal = files2bag(workDirNormal);
bagsAbnormal = files2bag(workDirAbnormal);
%%
[frameIdxNormal, TimeNormal, TNormal] = mapFrames(bagsNormal);
[frameIdxAbnormal, TimeAbnormal, TAbnormal] = mapFrames(bagsAbnormal);
%%
save temp.mat TNormal TAbnormal -mat
%%
load temp.mat
% Adding derivatives of some values?
tableNormalMagDer = addDerivative(TNormal, 'Mag', {'X', 'Y', 'Z'});
tableAbnormalMagDer = addDerivative(TAbnormal, 'Mag', {'X', 'Y', 'Z'});

tableNormal = removevars(tableNormalMagDer, {'MagX', 'MagY', 'MagZ'});
tableAbnormal = removevars(tableAbnormalMagDer, {'MagX', 'MagY', 'MagZ'});

%%
save tables.mat tableNormal tableAbnormal frameIdxNormal frameIdxAbnormal TimeNormal TimeAbnormal -mat
%% Modeling

clear; close all; clc;
load('tables.mat')
%% Feature generation

X_normal = tableNormal{:,:};
X_abnormal = tableAbnormal{:,:};

% add lookBack
kLast = 3;
X_normal = lookBack(X_normal, kLast);
X_abnormal = lookBack(X_abnormal, kLast);

% X = [X_normal; X_abnormal];
%% Train, CV, Test data split

splitPcg = [50, 25];
[X_normal_train, X_normal_cv, X_normal_test, idx_normal_train, idx_normal_cv, idx_normal_test] = splitData(X_normal, splitPcg);
[X_abnormal_train, X_abnormal_cv, X_abnormal_test, idx_abnormal_train, idx_abnormal_cv, idx_abnormal_test] = splitData(X_abnormal, splitPcg);

X_train = [X_normal_train; X_abnormal_train];
X_cv = [X_normal_cv; X_abnormal_cv];
X_test = [X_normal_test; X_abnormal_test];

M = size(X_train, 1);
M_normal = size(X_normal_train, 1);
M_abnormal = size(X_abnormal_train, 1);
%% Feature mapping (scaling and normalization)

mu = 0; %mean(X_train);
% Sigma = cov(X_train);
scalingFactor = 10^(-5);

X_train_mapped = (X_train - mu)* scalingFactor;
%% PCA

[coeff,score,latent,tsquared,explained,mu_pca] = pca(X_train_mapped);

varianceRetained = 99;
numComponents = find(cumsum(explained) >= varianceRetained, 1);

disp(['Number of principal components with ' int2str(varianceRetained) '% variance retained: '...
    int2str(numComponents) '/' int2str(size(X_train, 2))]);

U = coeff(:,1:numComponents);

Z = (X_train_mapped - mu_pca)* U;

save Z.mat Z -mat
%% Training data vizualization

y = [zeros(M_normal,1); ones(M_abnormal,1)];
figure
    gscatter(Z(:,1), Z(:,2), y, 'br', 'o+');
    legend('Normal recordings','Recordings with abnormalities');
    title(['Data projected on first 2 principal components, ' int2str(sum(explained(1:2))) '% var. ret.'])
    axis square
%% Gaussian Mixture Model

load Z.mat
numClasses = 2;
max_eval = 0;
Iterations = 1000;
for Lambda = 0.0001 : 0.005 :0.15
    disp(['Lambda = ' num2str(Lambda)]);
    [GMM, labelName, normal, abnormal, middle] = gmmFit(Z, numClasses, Lambda, Iterations);
    
    %% Results on CV set
    %% Feature normalization and reduction
    Z_cv = ((X_cv - mu)*scalingFactor - mu_pca) * U;
    
    %% Predict
    P_cv = posterior(GMM, Z_cv);
    [~, y_cv] = max(P_cv, [], 2);
           
    % data for silhouette analysis
    D_cv = silhouetteData(GMM, Z_cv, normal);
    
    %% Silhouette analysis
    s = silhouette(D_cv, y_cv);
    
    % silhouette evaluation
    w_normal = 40;
    thresh = 0; %-0.1;
    eval = silhouetteEval(s, y_cv, normal, w_normal, thresh);
    
    disp(['Evaluation (bigger is better): ' num2str(eval)]);
    
    if eval > max_eval
        max_eval = eval;
        GMM_opt = GMM;
        P_opt = P_cv;
        D_opt = D_cv;
        y_opt = y_cv;
        Lambda_opt = Lambda;
        normal_opt = normal;
        abnormal_opt = abnormal;
        labelName_opt = labelName;
    end
    
end

GMM = GMM_opt;
P_cv = P_opt;
D_cv = D_opt;
y_cv = y_opt;
Lambda = Lambda_opt;
normal = normal_opt;
abnormal = abnormal_opt;
labelName = labelName_opt;
% Visualization on CV set

% Silhouettes
[s, h] = silhouetteVisualization(D_cv, y_cv, numClasses, normal, abnormal, labelName, Lambda);
%%
norm(GMM.mu(abnormal,:)) - norm(GMM.mu(normal,:))
gmm2dVisualization(GMM, Z_cv, y_cv, labelName, Lambda);
%% Evaluate on test set

disp(['Optimal Lambda is:' num2str(Lambda_opt)]);

% feature maping and dim. reduction
Z_test = ((X_test - mu)*scalingFactor - mu_pca) * U;

% prediction
P_test = posterior(GMM, Z_test);
[~, y_test] = max(P_test, [], 2);

% data for silhouette analysis
D_test = silhouetteData(GMM, Z_test, normal);

%% Silhouette analysis
s = silhouette(D_test, y_test);

% silhouette evaluation
eval = silhouetteEval(s, y_test, normal, w_normal, thresh);
disp(['Evaluation on test set (bigger is better): ' num2str(eval)])
%% Visualize predicitons

% Silhouettes
[s, h] = silhouetteVisualization(D_test, y_test, numClasses, normal, abnormal, labelName, Lambda);
%%
gmm2dVisualization(GMM, Z, y, labelName, Lambda);
% gmm2dVisualization(GMM, Z_test, y_test, labelName, Lambda);
%% Some statistics

disp(['For Lambda:' num2str(Lambda)]);

M_pred = [length(find(y_test == normal)), length(find(y_test == abnormal))];
M_test = [length(X_normal_test), length(X_abnormal_test)];

M_pred_conf = [length(find(y_test(1:M_test(1)) == normal)), length(find(y_test(1:M_test(1)) == abnormal));...
               length(find(y_test(M_test(1)+1:end) == normal)), length(find(y_test(M_test(1)+1:end) == abnormal))];

disp(['Num samples normal: ' int2str(M_test(1))])
disp(['Num samples abnormal: ' int2str(M_test(2))])

disp(['Num predicted normal: ' int2str(M_pred(normal))])
disp(['Num predicted abnormal: ' int2str(M_pred(abnormal))])

disp(['Num predicted normal from normal set: ' int2str(M_pred_conf(1,1))])
disp(['Num predicted abnormal from normal set: ' int2str(M_pred_conf(1,2))])
disp(['Num predicted normal from abnormal set: ' int2str(M_pred_conf(2,1))])
disp(['Num predicted abnormal from abnormal set: ' int2str(M_pred_conf(2,2))])
%% Time visualization

n_normal_normal = idx_normal_test(y_test(1:M_test(1)) == normal);
n_normal_abnormal = idx_normal_test(y_test(1:M_test(1)) == abnormal);
n_abnormal_normal = idx_abnormal_test(y_test(M_test(1)+1:end) == normal);
n_abnormal_abnormal = idx_abnormal_test(y_test(M_test(1)+1:end) == abnormal);

n_normal = kLast + [n_normal_normal n_normal_abnormal];
n_abnormal = kLast + [n_abnormal_normal n_abnormal_abnormal];

t_normal = TimeNormal{n_normal,1};
t_abnormal = TimeAbnormal{n_abnormal,1};

y_normal = [zeros(M_pred_conf(1,1), 1); ones(M_pred_conf(1,2), 1)];
y_abnormal = [zeros(M_pred_conf(2,1), 1); ones(M_pred_conf(2,2), 1)];

figure
    subplot(1,2,1)
        stem(t_normal, y_normal);
        title('Normal test data');
        xlabel('t[s]')
    
    subplot(1,2,2)
        stem(t_abnormal, y_abnormal);
        title('Abnormal test data');
        xlabel('t[s]')
    
    sgtitle('Abnormalities in time. 0 - normal, 1 - abnormality detected.');
%% Saving the model

save GMM.mat GMM mu scalingFactor mu_pca U -mat
%