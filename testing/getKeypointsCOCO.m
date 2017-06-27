function predictions = getKeypointsCOCO(I, rect_roi, net, param)
% Given image, bbox, network and parameters, this method finds
% keypoint location and returns it as [x1, y1, v1,...].
% It can search through mutiple scales as defined in params (look at demo).

% COCO limb names
    % 0) nose
    % 1) left_eye 3) left_ear  5) left_shoulder 7) left_elbow 9) left_wrist 11) left_hip 13) left_knee 15) left_ankle
    % 2) right_eye 4) right_ear 6) right_shoulder 8) right_elbow 10) right_wrist 12) right_hip 14) right_knee 16) right_ankle
    labels = {'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',...
              'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'};

    center_person = [rect_roi(1) + rect_roi(3)/2, rect_roi(2) + rect_roi(4)/2];

    % set proper scale
    self_scale = rect_roi(4) / param.boxSize;
    scale_abs = param.target_scale / self_scale;
    
    % go through multiple scales
    results = cell(1, numel(param.scaleSearch));
    global results_mat;
    results_mat = zeros(size(I,2), size(I,1), param.numParts + 1, numel(param.scaleSearch));
    pad = cell(1, numel(param.scaleSearch));
    for i = 1:numel(param.scaleSearch)
        if param.DEBUG
            disp(strcat('Going through scale: ', num2str(param.scaleSearch(i))));
        end
        scale = scale_abs * param.scaleSearch(i);
        I_s = imresize(I, scale);
        center_person_s = center_person * scale;
        rect_roi_s = rect_roi * scale;
        [I_cpm, pad{i}] = padAround(I_s, param.boxSize, center_person_s, param.padValue);
        
        % prepare input for CPM
        I_cpm_input = preprocess(I_cpm, 0.5, param);

        % get results
        result = net.forward({single(I_cpm_input)});
        results{i} = result{1};
        
        pool_time = size(I_cpm, 1) / size(results{i}, 1);
        results{i} = imresize(results{i}, pool_time);
        results{i} = resizeIntoScaledImg(results{i}, pad{i});
        results{i} = imresize(results{i}, [size(I, 2) size(I, 1)]);
        results_mat(:,:,:,i) = results{i};
        
        % replace original image with scaled one
        % plot also CPM input (368x368)
        if param.DEBUG == true
            figure(1); clf;
            imshow(I_cpm)
            hold on;
            plot(center_person_s(1), center_person_s(2), 'r*');
            hold on;
            rectangle('Position', rect_roi_s, 'EdgeColor', 'green', 'LineWidth', 2);
            input('Press key to continue...');
        end
    end

    % summing up scores
    result = zeros(size(I,2), size(I,1), param.numParts);
    for i = 1:param.numParts
        result(:,:,i) = max(results_mat(:,:,i,:), [], 4);
    end
    result = permute(result, [2 1 3]);
    
    predictions = [];
    for i = 1:size(result, 3)
        [max_X, max_Y, score] = findMaximum(result(:,:,i));
        if score >= param.threshVisible
            visible = 1;
        else
            visible = 0;
        end
        predictions = [predictions, max_Y, max_X, visible, score];
    end
end


% Helper functions
function [x,y,score] = findMaximum(map)
    [score,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
end

function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    score = permute(score, [2 1 3]);
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
    score = permute(score, [2 1 3]);
end

function [img_padded, pad] = padAround(img, boxsize, center, padValue)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:), [1 pad(2) 1])*0 + padValue;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:), [1 pad(4) 1])*0 + padValue;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];

    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed
end

function img_out = preprocess(img, mean, param)
    img_out = double(img)/256;  
    img_out = double(img_out) - mean;
    img_out = permute(img_out, [2 1 3]);
    
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.boxSize;
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, param.sigma_center);
    img_out(:,:,4) = centerMapCell{1};
end

function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);
end
