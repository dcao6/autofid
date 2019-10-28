function [coarseoutput,descdone,locationdone] = rfcoarsetest(out, rf, fidnum, descdone, locationdone)

load(rf);
string = sprintf('Mdl%d',fidnum);
model = eval(string);

unsorted = predict(model,descdone); 

sorted = sort(unsorted,1);
sorted_unique = unique(sorted);
offset = [];
for i = 0:size(sorted_unique,1)-1
topchoice = find(unsorted==sorted_unique(end-i));
offset = [offset; locationdone(topchoice,:)];
end
coarseoutput = offset;
coarseoutput = [coarseoutput,flip(sorted(end-(size(sorted,1)-1):end))];

end