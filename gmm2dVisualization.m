function gmm2dVisualization(GMM, Z, y, labelName, Lambda)
%GMMVISUALIZATION 2D visualization of GMM with two classes
%   Detailed explanation goes here
    numClasses = GMM.NumComponents;

    pdfs = cell(numClasses, 1);
    for i = 1 : numClasses
        pdfs{i} = @(x1,x2) reshape(mvnpdf([x1(:) x2(:)], GMM.mu(i, 1:2), GMM.Sigma(1:2,1:2,i)), size(x1));    
    end
    
    figure
        for i = 1 : numClasses
            subplot(1,numClasses, i);
            gscatter(Z(:,1), Z(:,2), y); %, 'br', 'o+');
            g = gca;
            hold on
            fcontour(pdfs{i},[g.XLim g.YLim])
            
            xlim(GMM.mu(i,1) + 3*sqrt(GMM.Sigma(1,1, i))*[-1 1])
            ylim(GMM.mu(i,2) + 3*sqrt(GMM.Sigma(2,2, i))*[-1 1])
            
            title(['Fitted Gaussian ' labelName{i}])
            legend(labelName)
            hold off
        end
        sgtitle(['Lambda = ' num2str(Lambda)]);
    
    figure
        for i = 1 : numClasses
            subplot(1,numClasses, i);
            gscatter(Z(:,1), Z(:,2), y); %, 'br', 'o+');
            g = gca;
            hold on
            fsurf(pdfs{i},[g.XLim g.YLim])
            title(['Fitted Gaussian ' labelName{i}])
            legend(labelName)
            view(45, 15);
            axis fill
            hold off
        end      
        sgtitle(['Lambda = ' num2str(Lambda)]);
end

