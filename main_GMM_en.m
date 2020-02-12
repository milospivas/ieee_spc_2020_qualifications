%% IEEE Signal Processing Cup 2020 - Qualifications
% % 
% Team 2 : GMM
%% Report
% Defining the problem
% 5 normal and 5 abnormal recordings are given as ROS .bag files.
% 
% The task is to implement models which will give a prediction wheter each sample 
% in the recording is normal/abnormal based on measurements from .bag files.
% 
% Visualization:
% Images were extracted from "/pylon_camera_node/image_raw" topic, rotated 180 
% degrees and saved as video files with 4fps framerate.
% 
% From extracted videos we see that the recordings come from some UAV, where 
% the normal recordings were made during stable flight without fast movement while 
% abnormal recordings were made during very unstable flight with sudden movements.
% 
% % Preanalysis
% Every recording has around hundred mesurements but we don't have class labels.
% 
% => We can't use classic classification.
% 
% => Possible solution: Gaussian Mixture Model
% 
% % Measurement analysis
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
% Their derivatives are angular velocity, linear acceleration (already available 
% from IMU), and derivatives of magnetic field and compass heading (which can 
% be calculated from measurements), labeled as:
% 
% "IMUAngularVelocity", "IMULinearAcceleration", "MagneticFieldDerivative", 
% "compassHdgDerivative".
% 
% First three quantities are 3D, the fourth one is scalar, so the feature vectore 
% is 10D.
% 
% % Feature engineering
% TODO write
% Gaussian Mixture Model
% TODO write
% 
% % Overview of functions:

%     addDerivative      - calculates derivatives of selected columns of the table and adds the derivative columns.
%     bag2table          - Extracts '/mavros/imu/data', '/mavros/imu/mag', '/mavros/global_position/local'
%                           and '/mavros/global_position/compass_hdg' Topics from ROS .bag file and turns the data into a table.
%     bag2video          - Extracts images from ROS .bag file and saves the stream as video with given parameters.
%     bagCell2table      - Applies bag2table() on bag objects from given cell array and joins the results in one table.
%     bags2video         - Applies bag2video on all bag files from a given directory, returns a cell array of video output objects.
%     compassCell2table  - Converts a cell array of ROS bag 'std_msgs/Float64' type message into a table.
%     files2bag          - Loads all .bag files from given directory into a cell array of bag objects.
%     featureTable       - Data extraction. Applies bag2table() on bag objects from a given cell array,
%                         adds derivatives of magnetic field and compass heading columns, and extracts relevant columns.
%     imu2table          - Converts a ROS bag 'sensor_msgs/Imu' type message into a table.
%     imuCell2table      - Applies imu2table() on elements of a cell array of 'sensor_msgs/Imu' type messages and joins them into a table.
%     mag2table          - Converts ROS bag 'sensor_msgs/MagneticField' type messages into a table.
%     magCell2table      - Applies mag2table() on elements of a cell array of 'sensor_msgs/MagneticField' type messages and joins them into a table.
%     main_GMM_en            - Main program.
%     odom2table         - Converts ROS bag 'nav_msgs/Odometry' type messages into a table.
%     odomCell2table     - Applies odom2table() on elements of a cell array of 'nav_msgs/Odometry' type messages and joins them into a table.
%     plotData           - Plots chosen columns from a pair of tables as signals in time and their histograms.
%                         Used for measurement analysis.
%% 
% %% Implementation    
% Work directories for normal and abnormal data.

clear;

% change if needed
workDirNormal = '03_normal';
workDirAbnormal = '04_abnormal';
%% The structure of data
% Loading .bag files, selecting relevant measurements and converting into tables.

bagsNormal = files2bag(workDirNormal);
bagsAbnormal = files2bag(workDirAbnormal);
% Data analysis on joint data from all recordings and choosing predictor features.

jointTableNormal = bagCell2table(bagsNormal);
jointTableAbnormal = bagCell2table(bagsAbnormal);

% close all
Vector3Suffixes = ["X" "Y" "Z"];
QuaternionSuffixes = ["X" "Y" "Z" "W"];

plotData(jointTableNormal, jointTableAbnormal, "odomPosition", Vector3Suffixes, figure(1), figure(2));
% doesn't look useful

plotData(jointTableNormal, jointTableAbnormal, "odomOrientation", QuaternionSuffixes, figure(3), figure(4));
% high frequency oscillations on the abnormal data
% first derivative would be a very useful feature
% angular velocity? 

plotData(jointTableNormal, jointTableAbnormal, "odomTwistLinear", Vector3Suffixes, figure(5), figure(6));
% looks like first derivative would be very useful
% linear acceleration?

plotData(jointTableNormal, jointTableAbnormal, "IMUOrientation", QuaternionSuffixes, figure(7), figure(8));
% this looks identical to odomOrientation

plotData(jointTableNormal, jointTableAbnormal, "IMUAngularVelocity", Vector3Suffixes, figure(9), figure(10));
% just as anticipated, abnormals have much higher variance

plotData(jointTableNormal, jointTableAbnormal, "IMULinearAcceleration", Vector3Suffixes, figure(11), figure(12));
% just as anticipated, abnormals have much higher variance

plotData(jointTableNormal, jointTableAbnormal, "MagneticField", Vector3Suffixes, figure(13), figure(14));
% high frequency oscillations on the abnormal data
% first derivative would be a useful feature

plotData(jointTableNormal, jointTableAbnormal, "compassHdg", "", figure(15), figure(16));
% high frequency oscillations on the abnormal data
% first derivative would be a useful feature

jointTableNormalAddDer = addDerivative(jointTableNormal, "MagneticField", Vector3Suffixes);
jointTableAbnormalAddDer = addDerivative(jointTableAbnormal, "MagneticField", Vector3Suffixes);

jointTableNormalAddDer = addDerivative(jointTableNormalAddDer, "compassHdg");
jointTableAbnormalAddDer = addDerivative(jointTableAbnormalAddDer, "compassHdg");

plotData(jointTableNormalAddDer, jointTableAbnormalAddDer, "MagneticFieldDerivative", Vector3Suffixes, figure(17), figure(18));
% just as anticipated, abnormals have much higher variance

plotData(jointTableNormalAddDer, jointTableAbnormalAddDer, "compassHdgDerivative", "", figure(19), figure(20));
% just as anticipated, abnormals have much higher variance
%% Modeling
% Extracting data from bags

% tableNormal = featureTable(bagsNormal);
% tableAbnormal = featureTable(bagsAbnormal);
% save tables.mat tableNormal tableAbnormal -mat

clear; close all;
load('tables.mat')

% tableNormal = [tableNormal, table(zeros(height(tableNormal),1), 'VariableNames', "y")];
% tableAbnormal = [tableAbnormal, table(ones(height(tableAbnormal),1), 'VariableNames', "y")];
% Feature generation

X_normal = tableNormal{:,:};
X_abnormal = tableAbnormal{:,:};

% add lookBack
kLast = 4;
X_normal = lookBack(X_normal, kLast);
X_abnormal = lookBack(X_abnormal, kLast);

X = [X_normal; X_abnormal];
% Train, CV, Test data split

splitPcg = 70;
[X_normal_train, X_normal_test] = splitData(X_normal, splitPcg);
[X_abnormal_train, X_abnormal_test] = splitData(X_abnormal, splitPcg);

X_train = [X_normal_train; X_abnormal_train];
X_test = [X_normal_test; X_abnormal_test];

M = size(X_train, 1);
M_normal = size(X_normal_train, 1);
M_abnormal = size(X_abnormal_train, 1);

% Feature scaling and normalization

mu = mean(X_train);
sigma = std(X_train);
scalingFactor = 1/4 * 10^(-5);

X_train = (X_train - mu)*scalingFactor;

% PCA

[coeff,score,latent,tsquared,explained,mu] = pca(X_train);

varianceRetained = 99;
numComponents = find(cumsum(explained) >= varianceRetained, 1);

disp(['Number of principal components with ' int2str(varianceRetained) '% variance retained: '...
    int2str(numComponents) '/' int2str(size(X_train, 2))]);

U = coeff(:,1:numComponents);

Z = X_train * U;
% Training data vizualization

y = [zeros(M_normal,1); ones(M_abnormal,1)];
figure
    gscatter(Z(:,1), Z(:,2), y, 'br', 'o+');
    legend('Normal recordings','Recordings with abnormalities');
    title(['Data projected on first 2 principal components, ' int2str(sum(explained(1:2))) '% var. ret.'])
    xlim(0.2*[-1 1])
    ylim(0.2*[-1 1])
    axis square
% Gaussian Mixture Model

numClasses = 2;
GMM = fitgmdist(Z, numClasses);

% these can be used for model evaluation
disp(['BIC: ' num2str(GMM.BIC)]);
disp(['AIC: ' num2str(GMM.AIC)]);

if trace(GMM.Sigma(:,:,1)) < trace(GMM.Sigma(:,:,2))
    labelName = {'normal'; 'abnormal'};    
else
    labelName = {'abnormal'; 'normal'};
end
% GMM Visualization

y = [zeros(M_normal,1); ones(M_abnormal,1)];
pdfs = {@(x1,x2) reshape(mvnpdf([x1(:) x2(:)], GMM.mu(1, 1:2), GMM.Sigma(1:2,1:2,1)), size(x1));...
        @(x1,x2) reshape(mvnpdf([x1(:) x2(:)], GMM.mu(2, 1:2), GMM.Sigma(1:2,1:2,2)), size(x1))};

figure
    for i = 1 : 2
        subplot(1,2, i);
        gscatter(Z(:,1), Z(:,2), y, 'br', 'o+');
        g = gca;
        hold on
        fcontour(pdfs(i),[g.XLim g.YLim])
        
        xlim(GMM.mu(i,1) + 3*sqrt(GMM.Sigma(1,1, i))*[-1 1])
        ylim(GMM.mu(i,2) + 3*sqrt(GMM.Sigma(2,2, i))*[-1 1])
        
        title(['Fitted Gaussian ' labelName{i}])
        legend('Actual normal','Actual abnormal')
        hold off
    end
   
figure
    for i = 1 : 2
        subplot(1,2, i);
        gscatter(Z(:,1), Z(:,2), y, 'br', 'o+');
        g = gca;
        hold on
        fsurf(pdfs(i),[g.XLim g.YLim])
        title(['Fitted Gaussian ' labelName{i}])
        legend('Actual normal','Actual abnormal')
        view(45, 15);
        axis fill
        hold off
    end
%% 
% % Results on test set
% Feature normalization and reduction

X_test = (X_test - mu) * scalingFactor;
Z_test = X_test * U;
% Predict

P = posterior(GMM, Z_test);
[~, y_test] = max(P, [], 2);
% Visualize predicitons

M_pred = [length(find(y_test == 1)), length(find(y_test == 2))];
M_test = [length(X_normal_test), length(X_abnormal_test)];

M_pred_conf = [length(find(y_test(1:M_test(1)) == 1)), length(find(y_test(1:M_test(1)) == 2));...
               length(find(y_test(M_test(1)+1:end) == 1)), length(find(y_test(M_test(1)+1:end) == 2))];

disp(['Num samples normal: ' int2str(M_test(1))])
disp(['Num samples abnormal: ' int2str(M_test(2))])

disp(['Num predicted ' labelName{1} ': ' int2str(M_pred(1))])
disp(['Num predicted ' labelName{2} ': ' int2str(M_pred(2))])

disp(['Num predicted ' labelName{1} ' from normal set: ' int2str(M_pred_conf(1,1))])
disp(['Num predicted ' labelName{2} ' from normal set: ' int2str(M_pred_conf(1, 2))])
disp(['Num predicted ' labelName{1} ' from abnormal set: ' int2str(M_pred_conf(2, 1))])
disp(['Num predicted ' labelName{2} ' from abnormal set: ' int2str(M_pred_conf(2, 2))])

disp(['Predicted ' labelName{1} ' from normal test set '])
figure
    gscatter(Z_test(:,1), Z_test(:,2), y_test, 'br', 'o+');
    xlim(0.15*[-1 1])
    ylim(0.15*[-1 1])
    axis square
    
    legend(['Predicted ' labelName{1}], ['Predicted ' labelName{2}]);
    title(['Data projected on first 2 principal components'])

% Saving the model

save GMM.mat GMM -mat
%