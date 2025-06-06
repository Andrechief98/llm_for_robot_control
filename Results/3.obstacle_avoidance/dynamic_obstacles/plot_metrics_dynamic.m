clc;
clear;
close all;

%% Definitions
folders = {'o3-mini/2_obs','o3-mini/4_obs','o3-mini/6_obs','o3-mini/8_obs'};
numFolders = length(folders);

% Preallocation vector for each metric
collisionRate   = zeros(numFolders,1);
successRate     = zeros(numFolders,1);
minObsDistMean  = zeros(numFolders,1);
pathLengthRatioMean  = zeros(numFolders,1);
smoothnessMean  = zeros(numFolders,1);
inferenceTimeMean = zeros(numFolders,1);

collisionStd   = zeros(numFolders,1);
successStd     = zeros(numFolders,1);
minObsDistStd  = zeros(numFolders,1);
pathLengthRatioStd  = zeros(numFolders,1);
smoothnessStd  = zeros(numFolders,1);
inferenceTimeStd= zeros(numFolders,1);

objectsPerFolder = zeros(numFolders,1);

timestep = 0.5;  % secondi

%% Loop on folders
for i = 1:numFolders
    folderName = folders{i};
    bagFiles = dir(fullfile(folderName, '*.bag'));
    numSimulations = length(bagFiles);
    
    % Vectors to save metrcis for each simulation
    collisions   = zeros(numSimulations,1);  % 1 if collision, 0 otherwise
    successes    = zeros(numSimulations,1);   % 1 if success, 0 otherwise
    minObsDists  = zeros(numSimulations,1);
    pathLengthRatios  = zeros(numSimulations,1);
    smoothnesses = zeros(numSimulations,1);
    inferenceTimes = zeros(numSimulations,1);
    
    for j = 1:numSimulations
        bagFilePath = fullfile(folderName, bagFiles(j).name);
        fprintf('Elaborazione file: %s\n', bagFilePath);
        
        %% Opening rosbag and topic selection
        bag = rosbag(bagFilePath);
        
        % Topic principali
        topic_models        = '/gazebo/model_states';
        topic_boundingBoxes = '/gazebo/bounding_boxes';
        topic_generatedPath = '/gptGeneratedPath';
        topic_callDuration  = '/callDuration';
        
        % Selezione dei topic
        bagSel_models        = select(bag, 'Topic', topic_models);
        bagSel_boundingBox   = select(bag, 'Topic', topic_boundingBoxes);
        bagSel_generatedPath = select(bag, 'Topic', topic_generatedPath);
        bagSel_callDuration  = select(bag, 'Topic', topic_callDuration);
        
        %% Reading messages
        data_models       = readMessages(bagSel_models, 'DataFormat', 'struct');
        data_boundingBox  = readMessages(bagSel_boundingBox, 'DataFormat', 'struct');

        % To speed up, I can read the first two messages
        data_generatedPath= readMessages(bagSel_generatedPath, 1:2, 'DataFormat', 'struct');
        data_inferenceTime = readMessages(bagSel_callDuration, 'DataFormat', 'struct');
        
        %% Extraction bounding box (initial)
        
        totalObjects = length(data_boundingBox{1}.Min);
        if j == 1
            objectsPerFolder(i) = totalObjects;
        end
        
        boxes = cell(totalObjects,1);
        for k = 1:totalObjects
            xmin = data_boundingBox{1}.Min(k).X;
            ymin = data_boundingBox{1}.Min(k).Y;
            xmax = data_boundingBox{1}.Max(k).X;
            ymax = data_boundingBox{1}.Max(k).Y;
            boxes{k} = [xmin, xmax, ymin, ymax];  % format: [xmin, xmax, ymin, ymax]
        end
        
        % Actors are elements from 1 to totalObjects-1
        actor_boxes = boxes(1:totalObjects-1);
        
        % The robot is the last element
        robot_box = boxes{totalObjects};
        robot_width  = robot_box(2) - robot_box(1);
        robot_height = robot_box(4) - robot_box(3);
        
        %% Extraction actors and robot states from topic "/gazebo/model_states"
        % Here, the first element (index 1) is the ground floor and it must
        % be discarded. Actors are elements from 2 to totalObjects-1.
        % The robot is the last element (index totalObjects).
        
        numActors = totalObjects - 2;  % excluding ground floor and robot
        
        actorStates = cell(numActors,1);
        actorPos_initial = zeros(numActors,2);
        actorOrientations = zeros(numActors,1);  % yaw for each actor
        

        for a = 1:numActors
            actorState = data_models{1}.Pose(a+1);
            actorStates{a} = actorState;
            actorPos_initial(a,:) = [actorState.Position.X, actorState.Position.Y];
            
            % Conversion from quaternione to yaw (ROS convention)
            qx = actorState.Orientation.X;
            qy = actorState.Orientation.Y;
            qz = actorState.Orientation.Z;
            qw = actorState.Orientation.W;
            euler_rotations = quat2eul([qw, qx, qy, qz]); % "Z-Y-X"
            yaw = euler_rotations(1);
            actorOrientations(a) = yaw;
        end
        
        % robot'state (last element)
        robotState = data_models{1}.Pose(end);
        robotPos_initial = [robotState.Position.X, robotState.Position.Y];

        
        %% Extracting actors' velocities
        % For each actor, we can read the first message from topic 
        % " /actor{i}/cmd_vel" and compute the norm of the velocity.

        actorSpeeds = zeros(numActors,1);
        for a = 1:numActors
            topic_actor = sprintf('/actor%d/cmd_vel', a);
            bagSel_actor = select(bag, 'Topic', topic_actor);
            data_actor = readMessages(bagSel_actor, 1, 'DataFormat', 'struct');
            vx_cmd = data_actor{1}.Linear.X;
            vy_cmd = data_actor{1}.Linear.Y; 
            actorSpeeds(a) = sqrt(vx_cmd^2 + vy_cmd^2);
        end
        
        % Computing actor's velocity components
        actorVels = zeros(numActors,2);
        for a = 1:numActors
            actorVels(a,1) = actorSpeeds(a) * cos(actorOrientations(a));
            actorVels(a,2) = actorSpeeds(a) * sin(actorOrientations(a));
        end
        
        %% Extracting LLM's generated path (for the robot)
        poses = data_generatedPath{1}.Poses;
        numPoints = length(poses);
        originalPath = zeros(numPoints,2);
        for p = 1:numPoints
            originalPath(p,1) = poses(p).Position.X;
            originalPath(p,2) = poses(p).Position.Y;
        end
        
        % Linear interpolation to obtain the entire trajectory (considering
        % also the point between two consecutive LLM's generated points)
        consideredTimestep = 0.5;
        newTimestep = 0.1;

        t_original = (0:numPoints-1)*consideredTimestep;
        t_new = t_original(1):newTimestep:t_original(end);
        
        generatedPath = interp1(t_original, originalPath, t_new, 'linear');
        newNumPoints = length(generatedPath);
        
        %% Extraction Inference Time
        durationStr = data_inferenceTime{1}.Data;
        inferenceTimeVal = str2double(durationStr);
        inferenceTimes(j) = inferenceTimeVal;
        
        %% Computing future actors' positions

        actorPos_future = zeros(numActors, newNumPoints, 2);  % [actor, timestep, coordinates]
        
        for a = 1:numActors
            for p = 1:newNumPoints
                dt = (p-1)*newTimestep;
                actorPos_future(a, p, :) = actorPos_initial(a,:) + dt * actorVels(a,:);
            end
        end
        
        %% Collision detection 
        % For each point of the robot's trajectory, we determine
        % the actual oriented bounding box of both robot and all actors to
        % detect possible collisions 
        collision = false;
        minDistsPoints = zeros(newNumPoints,1);
        
        for p = 1:newNumPoints
            % Computing the oriented bounding box of the robot
            currRobotPos = generatedPath(p,:);
            if p < newNumPoints
                theta_robot = atan2(generatedPath(p+1,2)-generatedPath(p,2), generatedPath(p+1,1)-generatedPath(p,1));
            end
            local_points_robot = [ -robot_width/2,  robot_width/2,  robot_width/2, -robot_width/2, -robot_width/2;
                                   -robot_height/2, -robot_height/2,  robot_height/2,  robot_height/2, -robot_height/2 ];
            R_robot = [
                cos(theta_robot) -sin(theta_robot); 
                sin(theta_robot) cos(theta_robot)
                ];

            robot_corners = R_robot * local_points_robot;
            robot_corners(1,:) = robot_corners(1,:) + currRobotPos(1);
            robot_corners(2,:) = robot_corners(2,:) + currRobotPos(2);
            
            poly_robot = polyshape(robot_corners(1,:), robot_corners(2,:));

            for a = 1:numActors
                % Computing the oriented bounding box of the considered actor
                actorPos_p = squeeze(actorPos_future(a, p, :))';
                theta_actor = actorOrientations(a);
                
                % Actor's bounding box initial dimensions
                actor_box_initial = actor_boxes{a};
                actor_width  = actor_box_initial(2) - actor_box_initial(1);
                actor_height = actor_box_initial(4) - actor_box_initial(3);
                local_points_actor = [ -actor_width/2,  actor_width/2,  actor_width/2, -actor_width/2, -actor_width/2;
                                       -actor_height/2, -actor_height/2,  actor_height/2,  actor_height/2, -actor_height/2 ];
                R_actor = [
                    cos(theta_actor) -sin(theta_actor); 
                    sin(theta_actor) cos(theta_actor)
                    ];

                actor_corners = R_actor * local_points_actor;
                actor_corners(1,:) = actor_corners(1,:) + actorPos_p(1);
                actor_corners(2,:) = actor_corners(2,:) + actorPos_p(2);
                
                % Collision detection using polyshape
                poly_actor = polyshape(actor_corners(1,:), actor_corners(2,:));
                
                if area(intersect(poly_robot, poly_actor)) > 0
                    collision = true;
                    break;
                end
            end
            
            if collision
                break;
            end
        end

        if collision
            fprintf("COLLISION in the considered simulation \n")
        else
            fprintf("no collision in the considered simulation \n")
        end
        collisions(j) = collision;

        %% Computing Minimum Obstacle Distances
        minDistsPos = zeros(newNumPoints,1);
        
        for p = 1:newNumPoints
            currRobotPos = generatedPath(p,:);
            dists = zeros(numActors,1);
            for a = 1:numActors
                actorPos_p = squeeze(actorPos_future(a,p,:))';
                dists(a) = norm(currRobotPos - actorPos_p);
            end
            minDistsPos(p) = min(dists);
        end

        minObsDists(j) = min(minDistsPos);
        
        %% Computing Path Length Ratio
        distsBetweenPoints = sqrt(sum(diff(generatedPath,1,1).^2,2));
        euclideanDistance = norm([10,10]-[0,0]);
        pathLengthRatios(j) = euclideanDistance/sum(distsBetweenPoints);
        
        %% Computing Smoothness
        if numPoints < 3
            smoothnesses(j) = 0;
        else
            angles = zeros(numPoints-2,1);
            for p = 2:(numPoints-1)
                v1 = generatedPath(p,:) - generatedPath(p-1,:);
                v2 = generatedPath(p+1,:) - generatedPath(p,:);
                if norm(v1)==0 || norm(v2)==0
                    angles(p-1) = 0;
                else
                    angles(p-1) = acos(dot(v1,v2)/(norm(v1)*norm(v2)));
                end
            end
            smoothnesses(j) = mean(abs(angles));
        end
        
        %% Checking goal
        tol = 1e-2;
        finalPos = generatedPath(end,:);
        reachedGoal = (abs(finalPos(1)-10) < tol) && (abs(finalPos(2)-10) < tol);
        
        %% Success: goal reached and no collision
        if reachedGoal && ~collision
            successes(j) = 1;
        else
            successes(j) = 0;
        end
        
    end 
    
    % Computing mean and std_dev for each metric
    collisionRate(i)  = mean(collisions);
    collisionStd(i)   = std(collisions);
    
    successRate(i)    = mean(successes);
    successStd(i)     = std(successes);
    
    minObsDistMean(i) = mean(minObsDists);
    minObsDistStd(i)  = std(minObsDists);
    
    pathLengthRatioMean(i) = mean(pathLengthRatios);
    pathLengthRatioStd(i)  = std(pathLengthRatios);
    
    smoothnessMean(i) = mean(smoothnesses);
    smoothnessStd(i)  = std(smoothnesses);
    
    inferenceTimeMean(i)= mean(inferenceTimes);
    inferenceTimeStd(i)= std(inferenceTimes);
    
    fprintf('Folder %s: # objects = %d, Collision Rate = %.2f, Success Rate = %.2f, minObsDist = %.2f, pathLengthRatio = %.2f, Smoothness = %.2f, inferenceTime = %.2f\n',...
        folderName, objectsPerFolder(i), collisionRate(i), successRate(i), minObsDistMean(i), pathLengthRatioMean(i), smoothnessMean(i), inferenceTimeMean(i));
end

%% Plot delle metriche
save("final_metrics_dynamic.mat")



%% Figure with (1 x n_metrics) subplots

clc
% clear
% load("final_metrics_dynamic.mat")
close all
models = "o3-mini";

f = figure('Units','normalized','Position',[0.1 0.1 0.8 0.5]);
t = tiledlayout(1,5, ...
    'TileSpacing','compact', ...
    'Padding','compact');

% Parameters for each subplot
config = {
  successRate,        [],             [0,1],    0:0.2:1,   'SR';
  inferenceTimeMean,  inferenceTimeStd,[0,100],    0:20:100,    'IT [s]';
  minObsDistMean,     minObsDistStd,  [0,10],     0:2:10,      'MOD [m]';
  pathLengthRatioMean,pathLengthRatioStd,[0,1.2],  0:0.2:1.2,   'PLR';
  smoothnessMean,     smoothnessStd,  [-0.1,0.1], -0.1:0.05:0.1,'SM [rad]'
};

x = objectsPerFolder - 1;

% Loop on the 5 subplots
for k = 1:5
    ax = nexttile(t);
    box(ax, 'on');  
    hold(ax,'on')
    
    y    = config{k,1};
    yerr = config{k,2};
    
    if isempty(yerr)
        h = plot(ax, x, y, '-s', ...
            'LineWidth',1.5, 'MarkerSize',10);
    else
        h = errorbar(ax, x, y, (yerr), '-s', ...
            'LineWidth',1.5, 'MarkerSize',10);
    end
    
    main = h(1);
    main.MarkerFaceColor = main.Color;
    
    ax.XLim     = [0,10];
    ax.XTick    = [2,4,6,8];
    ax.YLim     = config{k,3};
    ax.YTick    = config{k,4};
    ax.FontSize = 14;
    xlabel(ax,'n_{obs}', 'Interpreter','tex','FontSize',14);
    ylabel(ax,config{k,5},'FontSize',14);
    grid(ax,'on')
    hold(ax,'off')
end


lg = legend(models, 'Orientation','horizontal', 'Box','on', 'FontSize',14);

lg.Layout.Tile      = 'south';     
lg.Layout.TileSpan  = [1 5];       

%%
exportgraphics(f, "dynamic_metrics_plot.pdf", "Resolution", 1000)