function frames = collectFrames(bags)
%COLLECTFRAMES Extracts and sorts frames
%   Extracts frames from passed bag files and contcatenates them into one cell array

frames={};
for b = 1:length(bags)
   bag_frames=extractImages(bags{b});
   frames{length(frames)+1:length(frames)+length(bag_frames)}=bag_frames{1:end}; 
end

end

