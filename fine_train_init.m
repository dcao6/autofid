function fine_train_init()

% This function initializes parameters for haar-like features.

vacant = zeros(81,81,81);

% This section initializes random coordinates for haar-like features.
% Coordinates determine where in image to extract a cubic volume.
for i = 3:30
savedsamplearr(i).s = combvec(i:size(vacant,1),i:size(vacant,2),i:size(vacant,3))';
end

if isfile('idxtrial1k.mat')
    load('idxtrial1k.mat')
else
    for i = 3:30
savedsamplearr(i).idxa = randsample(size(savedsamplearr(i).s,1),1000);
savedsamplearr(i).idxb = randsample(size(savedsamplearr(i).s,1),1000);
    end
save('idxtrial1k.mat','savedsamplearr')
load('idxtrial1k.mat')
end

% This section initializes random widths for each corresponding coordinate
% determined above.
if isfile('width1k.mat')
    load('width1k.mat')
else
    w = randsample([3:30],1000,true);
    save('width1k.mat','w')
    load('width1k.mat')
end
end