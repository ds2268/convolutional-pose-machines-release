%% % Prepare results for COCO eval2014 dataset

results = [];

% settings
param.img_path = 'sample_image/roger.png';
param.deploy_path = '../training/prototxt/COCO/pose_deploy.prototxt';
param.model_path = '../training/prototxt/COCO/caffemodel/pose_train_iter_200000.caffemodel';
param.target_scale = 0.8;
param.sigma_center = 21;
param.boxSize = 368;
param.gpuId = 0;
param.padValue = 128;
param.scaleSearch = 0.8; %0.7:0.1:1.3; % fit training
param.DEBUG = false; % debug getKeypointsCOCO

% debug only this code
DEBUG = false;

% setup Caffe network
caffe.set_mode_gpu();
caffe.set_device(param.gpuId);
net = caffe.Net(param.deploy_path, param.model_path, 'test');

%% initialize COCO api (please specify dataType/annType below)
annTypes = { 'instances', 'captions', 'person_keypoints' };
dataType='val2014'; annType=annTypes{3}; % specify dataType/annType
annFile=sprintf('../dataset/COCO/annotations/%s_%s.json',annType,dataType);
coco=CocoApi(annFile);

%% get all images containing persons
catIds = coco.getCatIds('catNms','person');
imgIds = coco.getImgIds('catIds',catIds);

%% load and display image
cnt = 0;
for i = 1:numel(imgIds)
   img = coco.loadImgs(imgIds(i));
   I = imread(sprintf('../dataset/COCO/%s/%s',dataType,img.file_name));
   
   % grayscale image - skip?
   if size(I, 3) == 1
       I = cat(3, I, I, I);
   end
   
   annIds = coco.getAnnIds('imgIds',imgIds(i),'catIds',catIds,'iscrowd',[]);
   
   if DEBUG
       figure(1);clf;
       imshow(I);
   end
   
   for j = 1:numel(annIds)
       ann = coco.loadAnns(annIds(j));
       if ann.num_keypoints == 0
           continue;
       end
       
       keypoints = getKeypointsCOCO(I, ann.bbox, net, param);
       results = [results, struct('image_id', imgIds(i), 'category_id', catIds,...
                  'keypoints', keypoints, 'score', 1)];
       
       if DEBUG
           hold on;
           plot(keypoints(2:3:end), keypoints(1:3:end), 'g*');
           input('Press key to continue...');
       end
   end
   
   cnt = cnt + 1;
   disp(strcat('Finished processing image:', num2str(cnt), '/', num2str(numel(imgIds))));
end

s = gason(results);
fid = fopen('results.json', 'w');
fprintf(fid, s);
fclose(fid);

% clear Caffe network from GPU
caffe.reset_all()

