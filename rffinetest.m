function [fineoutput] = rffinetest(out, rf, fidnum, units, finalcand,savedsamplearr,w,it)

mu = [61,61,61];
load(rf);

localpredfine = finalcand;
original = out;
out = original(localpredfine(1)-60:localpredfine(1)+60,localpredfine(2)-60:localpredfine(2)+60,localpredfine(3)-60:localpredfine(3)+60); 

if it == 1
    coordsinit = combvec(45:4:77,45:4:77,45:4:77);
else
    coordsinit = combvec(58:64,58:64,58:64);
end
coords = coordsinit';

answerarr = [];

for i = 1:size(coords,1)
    
    conarr = [];
    saved = out(coords(i,1)-40:coords(i,1)+40,coords(i,2)-40:coords(i,2)+40,coords(i,3)-40:coords(i,3)+40);
    SS = integralImage3(saved);
    farr = [];
    
    for k = 1:1000
        
        savedsample = savedsamplearr(w(k)).s;
        idxa = savedsamplearr(w(k)).idxa;
        idxb = savedsamplearr(w(k)).idxb;
        coordsa = savedsample(idxa(k),:);
        coordsb = savedsample(idxb(k),:);
        [sR, sC, sP, eR, eC, eP] = deal(coordsa(1)-w(k)+1, coordsa(2)-w(k)+1, coordsa(3)-w(k)+1, coordsa(1), coordsa(2), coordsa(3));
        [sR2, sC2, sP2, eR2, eC2, eP2] = deal(coordsb(1)-w(k)+1, coordsb(2)-w(k)+1, coordsb(3)-w(k)+1, coordsb(1), coordsb(2), coordsb(3));
        
        regionSum = SS(eR+1,eC+1,eP+1) - SS(eR+1,eC+1,sP) - SS(eR+1,sC,eP+1) ...
        - SS(sR,eC+1,eP+1) + SS(sR,sC,eP+1) + SS(sR,eC+1,sP) ... 
        + SS(eR+1,sC,sP) -SS(sR,sC,sP);
        regionSum2 = SS(eR2+1,eC2+1,eP2+1) - SS(eR2+1,eC2+1,sP2) - SS(eR2+1,sC2,eP2+1) ...
        - SS(sR2,eC2+1,eP2+1) + SS(sR2,sC2,eP2+1) + SS(sR2,eC2+1,sP2) ... 
        + SS(eR2+1,sC2,sP2) -SS(sR2,sC2,sP2);
        
        f = regionSum - regionSum2;
        farr = [farr,f];

    end
con = farr;
conarr = [conarr,con];
answerarr = [answerarr;conarr];
end

keys = keypoint3D(coords(:,1:3),repelem(0.5,size(coordsinit,2)));
[desc, location] = extractSift3D(keys,out, units);

string = sprintf('Mdlit0');
model = eval(string);
unsorted = predict(model,[desc,answerarr]);

fineoutput = localpredfine - [61,61,61] + location;
fineoutput = [fineoutput,unsorted]; 
end