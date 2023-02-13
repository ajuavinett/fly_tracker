function [mean_velocity, std_velocity, all_velocity] = BIPN145_flytrack(diameter,frame_rate,varargin)

%% SCRIPT INFO
% A short script to open an image and calculate the position of a fruit fly
% against a light background.
% Written by Jeff Stafford. Some code and ideas from Dan
% Valente's FTrack suite were used, as well as the 'Division' layer mode
% formula from GIMP for background subtraction.
% Modified by A Juavinett for BIPN 145

%% INPUTS
% diameter is diameter of dish in centimeters
if  nargin == 0
    diameter = 4; %default diameter
    frame_rate = 60; %default framerate
end

height = diameter;
width = diameter;

%% OUTPUTS
% CSV with following columns:
% column 1 = Time (in seconds)
% column 2 = Fly 1 x position (in cm from left edge of furthest left ROI)
% column 3 = Fly 1 y position (in cm from absolute top of ROI)

% The size of the area you want to search (in pixels).
search_size = 20;

% The average pixel intensity in the search area must exceed this value to
% log a position and NOT skip the frame. Prevents random noise and other
% weird stuff from "becoming the fly." Essentially requires any given blob
% it detects to be above a certain size and intensity.
per_pixel_threshold = 1.5;

%% OPEN VIDEO
[video_name, pathname] = uigetfile({'*.avi;*.mj2;*.mpg;*.mp4;*.m4v;*.mov', ...
    'Video Files (*.avi;*.mj2;*.mpg;*.mp4;*.m4v;*.mov)'}, ...
    'Select the videos you would like to analyze...', 'MultiSelect','on');
disp(['Opening ', video_name, ', please wait.']);
%video_name = [pathname, video_name];
file_list = strcat(pathname,video_name);
if (isa(file_list,'char'))
    file_list = {file_list};
end
num_files = size(file_list,2);
disp(strcat(num2str(num_files), ' files selected for analysis.'));

for i = 1:num_files
    if num_files == 1
        vr = VideoReader([pathname,video_name]);
    else
        vr = VideoReader([pathname,video_name{i}]);
    end
    resolution = [vr.Width vr.Height];
    nfrm_movie = floor(vr.Duration * vr.FrameRate) - 1;
    
    %% DEFINE REGION OF INTEREST(ROI)
    disp('Click and drag to define a rectangular region of interest, double-click to proceed.');
    figure('name', 'ROI select'), imshow(read(vr, 1));
    ROI_select = imrect;
    ROI = wait(ROI_select); %ROI takes form of [xmin ymin width height]
    close gcf;
    
    %% CREATE BACKGROUND
    
    disp('Calculating image background.');
    
    % Pick a random set of 100 frames to create the background.
    bg_number = 100;
    randv = rand(bg_number,1);
    bg_idx = sort(round(randv * nfrm_movie));
    
    % Read each frame of the background and average them to create a background
    % image.
    bg_array = zeros(resolution(2), resolution(1), bg_number, 'uint8');
    bg_step = 0;
    while bg_step < bg_number
        bg_step = bg_step + 1;
        bg_frame = rgb2gray(read(vr, bg_idx(bg_step)));
        bg_array(:,:,bg_step) = bg_frame;
    end
    background =  uint8(mean(bg_array, 3));
    
    %% ANALYZE VIDEO AND SUBTRACT BACKGROUND
    
    disp('Calculating fly positions (this part takes a while...).');
    
    % re-define search parameters to make sense regardless of size
    threshold = (search_size)^2 * per_pixel_threshold;
    search_size = round(search_size / 2);
    
    %initialize arrays used to log position
    pos_array = zeros(nfrm_movie, 3);
    
    %process frames of video for fly
    waitDialog = waitbar(0, 'Calculating fly positions');
    for nofr = 1:nfrm_movie
        waitbar(nofr/nfrm_movie, waitDialog, ...
            ['Analyzing frame ', num2str(nofr), ' of ',num2str(nfrm_movie)]);
        
        % Extract image from video.
        frame_int = uint8(rgb2gray(read(vr, nofr)));
        
        % Subtract image background using GIMP's image division formula.
        frame_int = uint8((256 * double(frame_int))./(double(background) + 1));
        
        % Find the fly
        frame_crop = imcrop(frame_int, ROI);
        fr_position = flyFinder(frame_crop, search_size, threshold, true);
        % nofr - 1 is for zero based numbering, like the larva tracker
        pos_array(nofr,:) = [nofr - 1, fr_position];
    end
    close(waitDialog);
    
    %% PROCESS & OUTPUT DATA
    
    disp('Creating output.');
    
    %Convert position coordinates to real, meaningful positions (coordinates in
    %cm and time in seconds. Scale is in cm/pixels.
    xscale = width / ROI(3);
    yscale = height / ROI(4);
    
    %convert to coordinates in cm and frame number to seconds, add coordinate
    %offset for bottom fly
    corrected_array = [pos_array(:,1)/vr.FrameRate, ...
        pos_array(:,2) * xscale, ...
        (pos_array(:,3) * yscale)];
    
    skipped = sum(isnan(corrected_array(:,2)));
    disp([num2str(skipped), ' points were skipped out of ', num2str(nfrm_movie),' for fly 1.']);
    
    % Teleport filter. Removes spurious points where fly position teleports all
    % over the vial due to a false track.
    if true
        corrected_array = distFilter(corrected_array, 2);
    end
    if (true)
        corrected_array = interpolatePos(corrected_array, 2);
    end
    
    %% MANUAL (TEMPORARY) FIX TO TIME AXIS
    if frame_rate == 15
        corrected_array(:,1) = corrected_array(:,1)/4;
    end     
    %% PLOT FLY PATH
    figure('Name','Pathing map');
    x = corrected_array(:,2)';
    y = corrected_array(:,3)';
    z = zeros(size(x));
    col = corrected_array(:,1)';
    surface([x;x],[y;y],[z;z],[col;col],...
        'facecol','no',...
        'edgecol','interp',...
        'linew',2, ...
        'LineSmoothing','on');
    xlabel('X-coordinate (cm)', 'fontsize', 11);
    ylabel('Y-coordinate (cm)', 'fontsize', 11);
    axis('manual', 'equal');
    axis([0 width 0 height]);
    % inverts the y coordinates to match the video
    set(gca, 'Ydir', 'reverse');
    c = colorbar('Location','southoutside');
    c.Label.String = 'Time (s)';
    
    %% CALCULATE VELOCITY
    % initialize
    
    % How long is the assay (in seconds)? If one of the csv files is shorter
    % than this, defaults to the shorter time.
    total_time = length(x)/frame_rate;
    
    % Bin size for velocity calculations (in seconds). Velocity is calculated
    % once per "bin size" by comparing positions at the start and end. Cannot
    % be lower than the rate of data being analyzed (generally 0.5-1s).
    binSize = 1;
    
    % recalculate total time to total # of bins
    total_time = floor(total_time / binSize);
    
    % calculate mean velocity per second for each replicate
    
    seconds = floor(total_time);
    meanVel = zeros(seconds,1);
    meanVel(:) = NaN;
    larvaNum = 1;
    
    replicate = corrected_array;
    num_rows = size(replicate,1);
    
    % extend meanVel array if we need to
    colsToAdd = size(replicate,2) - 3;
    if (colsToAdd > 0)
        newSpace = zeros(seconds, colsToAdd);
        newSpace(:) = NaN;
        meanVel = horzcat(meanVel, newSpace);
    end
    
    % calc velocities
    dataRate = round(1 / replicate(2, 1)) * binSize;
    if (dataRate < 1)
        mexception = MException('larva_velocity:BadParam', ...
            'Error: binSize is smaller than minimum data rate.');
        throw(mexception);
    end
    velocity = zeros(seconds, 1);
    % velocity(:) = NaN; % i'm not sure why this was setting to NaN - ALJ
    
    for animal = 1:2:(size(replicate, 2) - 1)
        % calculate velocity/s for each animal
        for row = 1:dataRate:(size(replicate, 1) - dataRate)
            % the '10 * ' converts to mm/s (coordinates are in cm)
            velocity(floor(row/dataRate)+1) = 10 * pdist2( ...
                [replicate(row,animal+1), replicate(row,animal+2)], ...
                [replicate(row+dataRate,animal+1), replicate(row+dataRate,animal+2)]);
        end
        % convert from mm/bin to mm/s
        velocity = velocity / binSize;
        
        % ensure that number of values and total size of array is the same
        if (length(velocity) > total_time)
            meanVel(:,larvaNum) = velocity(1:total_time);
        else
            meanVel(:,larvaNum) = velocity;
        end
        larvaNum = larvaNum + 1;
    end
    
    % remove any absurd velocities that aren't actually possible
    if max(meanVel > 30)
        disp('Warning: You have absurdly high velocities in this recording suggesting that the tracking is not working well.')
        disp('Might I advise changing the selected ROI or re-recording your video?')
    end
    %meanVel(meanVel > 50) = NaN;
    %meanVel(meanVel < 0.4) = 0;
    
    % find last datapoint and cut array down to size
    lastIdx = zeros(size(meanVel,2),1);
    for col = 1:size(meanVel,2)
        lastNonNaN = find(~isnan(meanVel(:,col)),1,'last');
        if (~isempty(lastNonNaN))
            lastIdx(col) = find(~isnan(meanVel(:,col)),1,'last');
        else
            % change entire velocity to zero, the larva likely didn't move
            meanVel(:,col) = 0;
            lastIdx(col) = NaN;
        end
    end
    meanVel = meanVel(1:max(lastIdx), :);
    
    % create a first column of timepoint labels
    plotData = horzcat((0:binSize:(size(meanVel,1) - 1) * binSize)', meanVel);
    
    all_velocity(i,:) = plotData(:,2);
    
    %% PLOT DATA
    figure('Name','Fly velocity');
    colormap = jet(size(meanVel, 2));
    hold on;
    for col = 2:size(plotData, 2)
        plot(plotData(:, 1), plotData(:, col), ...
            'linew', 1.5);
    end
    hold off;
    axis([0 (plotData(end, 1) + binSize) 0 (max(meanVel(:)) * 1.5)])
    xlabel('Time (s)', 'fontsize', 11);
    ylabel('Average velocity (mm/s)', 'fontsize', 11);
    
    % legend(cleanLabels(file_list), 'location', 'NorthWest');
    
%% FINAL DATA SUMMARY FOR THIS FILE
mean_velocity = mean(all_velocity);
std_velocity = std(all_velocity);

disp('Mean velocity for this video is:')
disp(mean_velocity)
disp('Standard deviation of velocity for this video is:')
disp(std_velocity)    

end


%% COMBINED PLOTS
if num_files > 1
    figure('Name','All fly velocities')
    hold on;
    plot(all_velocity','LineWidth',2)
    xlabel('Time (s)', 'fontsize', 11);
    ylabel('Average velocity (mm/s)', 'fontsize', 11);
    if num_files == 2
        legend({'Fly 1','Fly 2'})
    elseif num_files == 3
        legend({'Fly 1','Fly 2','Fly 3'})
    elseif num_files == 4
        legend({'Fly 1','Fly 2','Fly 3','Fly 4'})
    elseif num_files == 4
        legend({'Fly 1','Fly 2','Fly 3','Fly 4','Fly 5'})
    end

    disp('Mean velocity ACROSS videos is:')
    mean_velocity_all = mean(mean(all_velocity,2));
    disp(mean_velocity_all)
    disp('Standard deviation (NOT error) of mean velocity ACROSS videos is:')
    SD_velocity_all = std(mean(all_velocity,2));
    disp(SD_velocity_all)      
end

%% WRITE VELOCITY DATA TO DISC
%
% [output_name,path] = uiputfile('.csv');
% if output_name ~= 0  % in case someone closes the file saving dialog
%     csvwrite(strcat(path,output_name), plotData);
% else
%     disp('File saving cancelled.')
% end
%

%% WRITE OUTPUT OF FLYTRACK_VIDEO
% [output_name,path] = uiputfile('.csv');
% output_array = corrected_array;
% if output_name ~= 0  % in case someone closes the file saving dialog
%     csvwrite(strcat(path,output_name), output_array);
% else
%     disp('File saving cancelled.')
% end
%
% return;
