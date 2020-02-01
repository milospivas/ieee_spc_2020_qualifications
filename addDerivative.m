function [T] = addDerivative(inputTable,  fieldBaseName, fieldSuffixes)
%ADDDERIVATIVE Add derivatives of the selected table fields to table.
%   Add derivatives of selected table fields to table.
%   Fields are selected with parameters fieldBaseName and fieldSuffixes
%   by concatenating every string from fieldSuffixes to fieldBaseName.

    if nargin < 3
        fieldSuffixes = "";
    end

    T = inputTable;
    fieldBaseNameDerivative = fieldBaseName + "Derivative";
    
    for s = fieldSuffixes
        T = [T, table(zeros(size(T, 1), 1), 'VariableNames', fieldBaseNameDerivative+s)];

        T{2:end, fieldBaseNameDerivative+s} = T{2:end, fieldBaseName+s} - T{1:end-1, fieldBaseName+s};
    end
end

