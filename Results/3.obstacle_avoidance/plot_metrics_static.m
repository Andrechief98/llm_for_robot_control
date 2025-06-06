clc;
clear;
close all;

%% Definitions
models = {'gpt-4o zero-shot','gpt-4o few-shot','DeepSeek V3 zero-shot','DeepSeek V3 few-shot','o3-mini'};  
numModels = length(models);

obsFolders = {'2_obs','4_obs','6_obs','8_obs'};
numObs = length(obsFolders);

collisionRate      = zeros(numObs, numModels);
collisionStd       = zeros(numObs, numModels);
successRate        = zeros(numObs, numModels);
successStd         = zeros(numObs, numModels);
inferenceTimeMean  = zeros(numObs, numModels);
inferenceTimeStd   = zeros(numObs, numModels);
minObsDistMean     = zeros(numObs, numModels);
minObsDistStd      = zeros(numObs, numModels);
pathLengthRatioMean     = zeros(numObs, numModels);
pathLengthRatioStd      = zeros(numObs, numModels);
smoothnessMean     = zeros(numObs, numModels);
smoothnessStd      = zeros(numObs, numModels);

objectsPerFolder   = zeros(numObs, numModels);

%% Loop on models and experiment (characterized by the number of obstacles)
for m = 1:numModels
    for i = 1:numObs
        % Building the path: es. 'o3-mini/2_obs'
        folderPath = fullfile(models{m}, obsFolders{i});
        bagFiles = dir(fullfile(folderPath, '*.bag'));
        numSimulations = length(bagFiles);
        
        % Vector preallocation for each simulation metrics
        collisions        = false(numSimulations,1);
        successes         = zeros(numSimulations,1);
        minObsDists     = zeros(numSimulations,1);
        pathLengthRatios  = zeros(numSimulations,1);
        smoothnesses      = zeros(numSimulations,1);
        inferenceTimes    = zeros(numSimulations,1);
        
        for j = 1:numSimulations
            bagFilePath = fullfile(folderPath, bagFiles(j).name);
            fprintf('Modello: %s, Cartella: %s, File: %s\n', models{m}, obsFolders{i}, bagFiles(j).name);

            %% Opening rosbag and topic selection
            bag = rosbag(bagFilePath);

            data_models        = readMessages(select(bag,'Topic','/gazebo/model_states'), 'DataFormat','struct');
            data_boundingBox   = readMessages(select(bag,'Topic','/gazebo/bounding_boxes'), 'DataFormat','struct');
            data_generatedPath = readMessages(select(bag,'Topic','/gptGeneratedPath'), 1:2, 'DataFormat','struct'); % To speed up, I can read the first two messages
            data_inferenceTime = readMessages(select(bag,'Topic','/callDuration'),'DataFormat','struct');
            

            %% Extraction generated trajectory
            poses = data_generatedPath{1}.Poses; % LLM's generated path
            numPoints = length(poses);
            generatedPath = zeros(numPoints,2);
            for p = 1:numPoints
                generatedPath(p,:) = [poses(p).Position.X, poses(p).Position.Y];
            end


            %% Extraction obstacle positions
            totalObjects = length(data_boundingBox{1}.Min);
            if j == 1
                objectsPerFolder(i,m) = totalObjects;
            end

            totalObstacles = totalObjects-1;
            
            obsPos = zeros(totalObstacles, 2);

            % Since we have static obstacles, we can extract the first
            % message to obtain positions of each obstacle
            obstaclesPositionsMsgs = data_models{1}.Pose(2:end-1);

            for obs_index = 1:totalObstacles
                obsPos(obs_index,:) = [obstaclesPositionsMsgs(obs_index).Position.X, obstaclesPositionsMsgs(obs_index).Position.Y];
            end



            %% Extraction bounding boxes 

            boxes = cell(totalObjects,1);
            for k = 1:totalObjects
                xmin = data_boundingBox{1}.Min(k).X;
                ymin = data_boundingBox{1}.Min(k).Y;
                xmax = data_boundingBox{1}.Max(k).X;
                ymax = data_boundingBox{1}.Max(k).Y;
                boxes{k} = [xmin, xmax, ymin, ymax];
            end

            % Separation robot's bounding box (last element) from obstacles' bounding boxes
            robot_box = boxes{end};
            robot_width  = robot_box(2) - robot_box(1);
            robot_height = robot_box(4) - robot_box(3);
            obs_boxes = boxes(1:end-1);

            %% Extraction inferenceTime
            inferenceTimes(j) = str2double(data_inferenceTime{1}.Data);
   

            %% Check collisions

            for p = 1:numPoints
                
                robotPos = generatedPath(p,:);
                q  = data_models{1}.Pose(end).Orientation; %quaternion
                eul         = quat2eul([q.W, q.X, q.Y, q.Z], 'ZYX');
                robotTheta  = eul(1);  % yaw
                
                obsBoxes = cell2mat(obs_boxes);
                
                % Check collision
                collision = checkCollision(robotPos, [robot_width, robot_height], robotTheta, obsBoxes);
                
                if collision == true
                    break
                end
            end
            


            %% Check goal
            tol = 1e-2;
            finalPos = generatedPath(end,:);
            goalPos = [10,10];
            reachedGoal = norm(goalPos - finalPos)<tol;

            if reachedGoal && ~collision
                successes(j) = 1;
            else
                successes(j) = 0;
            end

            %% Computing Minimum Obstacle Distance
            robot_obsDists = zeros(size(generatedPath,1),totalObstacles);

            for n_obs = 1:totalObstacles
                robot_obsDists(:,n_obs) = vecnorm(generatedPath-obsPos(n_obs,:),2,2);
            end

            minObsDists(j) = min(robot_obsDists,[],"All");

            
                
                
            
            %% Computing Path Length Ratio
            % Computing actual total length of the generated path
            distsBetweenPoints = sqrt(sum(diff(generatedPath,1,1).^2,2));
            actualPathLength = sum(distsBetweenPoints);
            
            % Computing the Euclidean distance between the first point (0,0)
            %  and the last point of the ideal trajectory (10,10) 
           
            euclideanDistance = norm([10,10]-[0,0]);

            % Computing Path Length Ratio
            pathLengthRatio = euclideanDistance / actualPathLength;
            
            % Save the value for each simulation
            pathLengthRatios(j) = pathLengthRatio;
            
            %% Computing Smoothness
            if size(generatedPath,1) < 3
                smoothnesses(j) = 0;
            else
                angles = zeros(numPoints-2,1);
                for p = 2:(numPoints-1)
                    v1 = generatedPath(p,:) - generatedPath(p-1,:);
                    v2 = generatedPath(p+1,:) - generatedPath(p,:);
                    if norm(v1)==0 || norm(v2)==0
                        angles(p-1) = 0;
                    else
                        angles(p-1) = acos(v1*v2'/(norm(v1)*norm(v2)));
                    end
                end
                smoothnesses(j) = mean(abs(angles));
            end
        end  

        collisionRate(i,m)     = mean(collisions);
        collisionStd(i,m)      = std(collisions);
        successRate(i,m)       = mean(successes);
        successStd(i,m)        = std(successes);
        inferenceTimeMean(i,m) = mean(inferenceTimes);
        inferenceTimeStd(i,m)  = std(inferenceTimes);
        minObsDistMean(i,m)  = mean(minObsDists);
        minObsDistStd(i,m)   = std(minObsDists);
        pathLengthRatioMean(i,m)    = mean(pathLengthRatios);
        pathLengthRatioStd(i,m)     = std(pathLengthRatios);
        smoothnessMean(i,m)    = mean(smoothnesses);
        smoothnessStd(i,m)     = std(smoothnesses);


        fprintf('Model %s, Folder %s: # n_{obs} = %d, Collision Rate = %.2f, Success Rate = %.2f, inferenceTime = %.2f, minObsDist = %.2f, PathLength = %.2f, Smoothness = %.2f\n',...
            models{m}, obsFolders{i}, objectsPerFolder(i, m), collisionRate(i, m), successRate(i, m), ...
            inferenceTimeMean(i, m), minObsDistMean(i, m), pathLengthRatioMean(i, m), smoothnessMean(i, m));
    end
end

save("final_metrics_static.mat")



%%
clc
clear
load("final_metrics_static.mat")

close all

% X-axis with number of obstacles obtained from folders names (e.g., "2_obs" -> 2)
x = zeros(numObs,1);
for i = 1:numObs
    parts = split(obsFolders{i}, '_');
    x(i) = str2double(parts{1});
end

% Settings for each metric (subplots)
yLimits = { [0, 1.2], [0, 120], [0, 6], [0, 1], [-0.2, 0.6] };
yTicks  = { [0.2, 0.4, 0.6, 0.8, 1, 1.2], [0, 20, 40, 60, 80, 100, 120], [0,2,4,6], [0, 0.2, 0.4, 0.6, 0.8, 1], [-0.2,0,0.2,0.4,0.6] };
xLimits = [0, 10];
xTicks  = [2, 4, 6, 8];

% Aggregate metrics
yData    = { successRate, inferenceTimeMean, minObsDistMean, pathLengthRatioMean, smoothnessMean };

% No bar error for Success Rate
errData  = { [], inferenceTimeStd, minObsDistStd, pathLengthRatioStd, smoothnessStd};

nMetrics = numel(yData);

% Colors for each model
colors = [
    0.0000, 0.4470, 0.7410; % blu: GPT-4o zero-shot
    0.8500, 0.3250, 0.0980; % orange: GPT-4o few-shot
    0.9290, 0.6940, 0.1250; % yellow: DeepSeek V3 zero-shot
    0.4940, 0.1840, 0.5560; % violet: DeepSeek V3 few-shot
    0.4660, 0.6740, 0.1880  % green: o3-mini
];

% Line for each model
lineStyles = {'-', '--', ':', '-.', '--'};

% Metrics labels
yLabels = {'SR', 'IT', 'PLR', 'MOD', 'SM'};


f = figure;

for r = 1:nMetrics
    subplot(1, nMetrics, r);
    hold on;

    for m = 1:numModels
        if r == 1
            % Only marker and line for Success Rate
            plot(x, yData{r}(:,m), lineStyles{m}, ...
                'Color', colors(m,:), ...
                'LineWidth', 2.5);
            plot(x, yData{r}(:,m), 's', ...
                'MarkerFaceColor', colors(m,:), ...
                'MarkerEdgeColor', colors(m,:), ...
                'MarkerSize', 10, ...
                'LineStyle', 'none');
        else
            % Other metrics (errorbar and different lines)
            errorbar(x, yData{r}(:,m), errData{r}(:,m), ...
                lineStyles{m}, ...
                'Color', colors(m,:), ...
                'LineWidth', 2.5, ...
                'Marker', 's', ...
                'MarkerFaceColor', colors(m,:), ...
                'MarkerEdgeColor', colors(m,:), ...
                'MarkerSize', 10);
        end
    end

    xlabel('n_{obs}', "FontSize", 12, "Interpreter", "tex");
    ylabel(yLabels{r}, "FontSize", 14); 
    set(gca, 'XLim', xLimits, 'XTick', xTicks, 'YLim', yLimits{r}, 'YTick', yTicks{r});
    grid on;
end


legend(models, 'Position', [0.9 0.2 0.05 0.6], 'Interpreter', 'none');


