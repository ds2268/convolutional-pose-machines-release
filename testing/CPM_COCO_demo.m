clear all;
close all;

% This is basic code to test learned model(COCO) on example image

% COCO limb names
% 0) nose
% 1) left_eye 3) left_ear  5) left_shoulder 7) left_elbow 9) left_wrist 11) left_hip 13) left_knee 15) left_ankle
% 2) right_eye 4) right_ear 6) right_shoulder 8) right_elbow 10) right_wrist 12) right_hip 14) right_knee 16) right_ankle
labels = {'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',...
          'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'};

% settings
param.img_path = 'sample_image/shihen.png';
param.deploy_path = '../training/prototxt/COCO/pose_deploy.prototxt';
param.model_path = '../training/prototxt/COCO/caffemodel_trained_100k_00008/pose_iter_100000.caffemodel';
param.target_scale = 0.8;
param.sigma_center = 21;
param.boxSize = 368;
param.gpuId = 0;
param.padValue = 128;

I = imread(param.img_path);

figure(1);
imshow(I);

% select ROI
rect_roi = getrect(1);
center_person = [(rect_roi(1) + rect_roi(1) + rect_roi(3))/2, (rect_roi(2) + rect_roi(2) + rect_roi(4))/2];

% set proper scale
self_scale = rect_roi(4) / param.boxSize;
scale_abs = param.target_scale / self_scale;
I_s = imresize(I, scale_abs);
center_person_s = center_person * scale_abs;
rect_roi_s = rect_roi * scale_abs;
%I_cpm = I_s(center_person_s(2)-param.boxSize/2+1:center_person_s(2)+param.boxSize/2, center_person_s(1)-param.boxSize/2:center_person_s(1)+param.boxSize/2-1, :);
[I_cpm, pad] = padAround(I_s, param.boxSize, center_person_s, param.padValue);

% replace original image with scaled one
% plot also CPM input (368x368)
figure(1); clf;
imshow(I_s)
hold on;
plot(center_person_s(1), center_person_s(2), 'r*');
hold on;
rectangle('Position', rect_roi_s, 'EdgeColor', 'green', 'LineWidth', 2);

figure(2); clf;
imshow(I_cpm);
size(I_cpm)

% prepare input for CPM
I_cpm_input = preprocess(I_cpm, 0.5, param);

% get results
caffe.set_mode_gpu();
caffe.set_device(param.gpuId);
net = caffe.Net(param.deploy_path, param.model_path, 'test');
result = net.forward({single(I_cpm_input)});
result = result{1};

% rescale heatmaps back to input size (368x368) % stride 8
result_368 = {};
I_plot_cpm = I_cpm;
I_plot_org = I;
results_org = {};
for i = 1:size(result, 3) - 1
    tmp = permute(result(:,:,i), [2, 1]);
    result_368{i} = imresize(tmp, 8);
    [max_X, max_Y] = findMaximum(result_368{i});
    I_plot_cpm = insertText(I_plot_cpm, [max_Y, max_X], labels(i),...
                 'FontSize',8,'TextColor','green', 'BoxOpacity', 0);
    I_plot_cpm = insertMarker(I_plot_cpm, [max_Y, max_X],'x','color', 'red', 'size', 3);
    results_org{i} = imresize(resizeIntoScaledImg(result_368{i}, pad), [size(I, 1) size(I, 2)]);
    
    [max_X, max_Y] = findMaximum(results_org{i});
    I_plot_org = insertText(I_plot_org, [max_Y, max_X], labels(i),...
                 'FontSize',16,'TextColor','green', 'BoxOpacity', 0);
    I_plot_org = insertMarker(I_plot_org, [max_Y, max_X],'x','color', 'red', 'size', 10);
end

% plot keypoints on CPM input and on original image
figure(3);clf;
imshow(I_plot_cpm);
figure(4);clf;
imshow(I_plot_org);


caffe.reset_all()

%% Helper functions
function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
end

function score = resizeIntoScaledImg(score, pad)
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2)));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2)));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2)));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4)));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
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