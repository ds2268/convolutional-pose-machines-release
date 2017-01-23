%% Evaluation code for COCO
% run COCO_benchmark_prepare to get results in json beforehand

%% select results type for demo (either bbox or segm)
type = {'segm','bbox','keypoints'}; type = type{3}; % specify type here
fprintf('Running demo for *%s* results.\n\n',type);

%% initialize COCO ground truth api
dataDir='../dataset/COCO'; prefix='instances'; dataType='val2014';
if(strcmp(type,'keypoints')), prefix='person_keypoints'; end
annFile=sprintf('%s/annotations/%s_%s.json',dataDir,prefix,dataType);
cocoGt=CocoApi(annFile);

%% initialize COCO detections api
resFile='results_100.json';
cocoDt=cocoGt.loadRes(resFile);

%% visualize gt and dt side by side
imgIds=sort(cocoGt.getImgIds()); 

% for visualization only - comment this if benchmarking
%imgIds=imgIds(1:100);
imgId = imgIds(randi(100)); img = cocoGt.loadImgs(imgId);
I = imread(sprintf('%s/val2014/%s',dataDir,img.file_name));
figure(1); subplot(1,2,1); imagesc(I); axis('image'); axis off;
annIds = cocoGt.getAnnIds('imgIds',imgId); title('ground truth')
anns = cocoGt.loadAnns(annIds); cocoGt.showAnns(anns);
figure(1); subplot(1,2,2); imagesc(I); axis('image'); axis off;
annIds = cocoDt.getAnnIds('imgIds',imgId); title('results')
anns = cocoDt.loadAnns(annIds); cocoDt.showAnns(anns);

%% load raw JSON and show exact format for results
fprintf('results structure have the following format:\n');
res = gason(fileread(resFile)); disp(res)

%% the following command can be used to save the results back to disk
if(0), f=fopen(resFile,'w'); fwrite(f,gason(res)); fclose(f); end

%% run COCO evaluation code (see CocoEval.m)
cocoEval=CocoEval(cocoGt,cocoDt,type);
cocoEval.params.imgIds=imgIds;
cocoEval.evaluate();
cocoEval.accumulate();
cocoEval.summarize();

%% generate Derek Hoiem style analyis of false positives (slow)
if(0), cocoEval.analyze(); end