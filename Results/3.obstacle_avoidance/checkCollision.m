function collision = checkCollision(robotPos, robotSize, robotTheta, obsBoxes)
    % robotPos    : [x, y]
    % robotSize   : [width, height]
    % robotTheta  : yaw angle in radians
    % obsBoxes    : N×4 matrix [xmin, xmax, ymin, ymax] per row

    % 1) Robot box corners in local frame
    w = robotSize(1)/2;
    h = robotSize(2)/2;
    localCorners = [-w, -h;
                     w, -h;
                     w,  h;
                    -w,  h];

    % 2) Rotate & translate to world frame
    R = [cos(robotTheta), -sin(robotTheta);
         sin(robotTheta),  cos(robotTheta)];
    worldCorners = (R * localCorners')' + robotPos;

    % 3) Build polyshape for robot
    robotPoly = polyshape(worldCorners(:,1), worldCorners(:,2));

    % 4) Check overlap with each obstacle
    collision = false;
    for k = 1:size(obsBoxes,1)
        xmin = obsBoxes(k,1); xmax = obsBoxes(k,2);
        ymin = obsBoxes(k,3); ymax = obsBoxes(k,4);
        obsCorners = [xmin, ymin;
                      xmax, ymin;
                      xmax, ymax;
                      xmin, ymax];
        obsPoly = polyshape(obsCorners(:,1), obsCorners(:,2));
        if overlaps(robotPoly, obsPoly)
            collision = true;
            return;
        end
    end
end
