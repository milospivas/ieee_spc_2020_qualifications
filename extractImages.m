function images = extractImages(bag)
%EXTRACTIMAGES Extracts images from bag file
%   Forms a set of all images from one bag file
bSel = select(bag,'Topic','/pylon_camera_node/image_raw');
normal_images = readMessages(bSel,'DataFormat','struct');
images = [];
for i=1:length(normal_images)
  image = vec2mat(normal_images{i,1}.Data,normal_images{i,1}.Width);
  image = imrotate(image,180);
  images = [images; {image}];
end

end

