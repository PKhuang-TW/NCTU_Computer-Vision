function [paths, labels] = LoadCSV(csv)
%     csv = '../csv/test.csv';
    f = fopen(csv);
    out = textscan(f, '%s%s', 'delimiter', ',');
    fclose(f);
    [paths, labels] = deal(out{:});
    paths = paths(2:end);  % Remove header
    labels = labels(2:end);  % Remove header
    paths = cell2vector(paths, 'str');  % Change cell into string array
    labels = cell2vector(labels, 'double');  % Change cell into double array
end
    
function vec = cell2vector(c, toType)
    vec = [];
    for i = 1:length(c)
        item = c(i);
        item = item{:};
        if strcmp(toType,'str')
            item = convertCharsToStrings(item);
        else
            item = str2num(item);
        end
        vec = [vec; item];
    end
end