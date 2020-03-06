function frames = collectFrames(bags)
%COLLECTFRAMES Extracts and sorts frames
%   Extracts frames from passed bag files and contcatenates them into one array

frames = [];
for b = 1:length(bags)
    bagSelect = select(bags{b},'Topic','/pylon_camera_node/image_raw');
    images = readMessages(bagSelect,'DataFormat','struct');
    for i = 1:length(images)
         image = vec2mat(images{i,1}.Data,images{i,1}.Width);
         image = imrotate(image,180);
         frames = [frames {image}];
    end
end

end

