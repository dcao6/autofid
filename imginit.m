function [descdone,locationdone] = imginit(out, units)

% Target coordinates.
coordsinit = combvec(31:2:size(out,1)-30,31:2:size(out,2)-30,31:2:size(out,3)-30);
coords = coordsinit';

% Extract SIFT features.
keys = keypoint3D(coords(:,1:3),repelem(0.5,size(coordsinit,2)));
[desc, location] = extractSift3D(keys,out, units);

% Output results.
descdone = desc;
locationdone = location;
end
