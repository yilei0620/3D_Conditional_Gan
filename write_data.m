
s = 701;
e = 800;
find_label = 5; %chair
count_num = 132;
rgb = [180 60 20];
for t = s:e
    labeltmp = labels(:,:,t);
    if find(labeltmp == find_label)
        labeltmp(labeltmp~=find_label) = 0;
        labeltmp(labeltmp~=0) = 1;
        labeltmp = logical(labeltmp);
        imgtmp = images(:,:,:,t);
        depthtmp = depths(:,:,t);
        kinp = 'c';
        figure(1)
        subplot(1,2,1)
        imshow(labeltmp)
        subplot(1,2,2)
        imshow(imgtmp)
        ksel = input('Nice Data?[y/n]','s');
        if ksel == 'n'
            close(1)
            continue
        end
        while kinp == 'c'
            [~, rect] = imcrop(imgtmp);
            Isave1 = imgtmp(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),1);
            [H,W] = size(Isave1);
            Isave2 = imgtmp(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),2);
            Isave3 = imgtmp(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),3);
            dsave = depthtmp(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
            lsave = labeltmp(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
            Isave1(lsave ~= 1) = 255;
            Isave2(lsave ~= 1) = 255;
            Isave3(lsave ~= 1) = 255;
            Isave = zeros(H,W,3);
            Isave(:,:,1) =  Isave1;
            Isave(:,:,2) =  Isave2;
            Isave(:,:,3) =  Isave3;
            imwrite(uint8(Isave),['img/' int2str(count_num) '.jpg'])
            
            dsave(lsave ~= 1) = 0;
            dtmp = dsave(lsave == 1);
            norm_min = min(dtmp(:));
            dtmp = dtmp - norm_min;
            norm_max = max(dtmp(:));
            dtmp = dtmp/norm_max;
            dsave(lsave == 1) = dtmp;
            imwrite(dsave,['depth/' int2str(count_num) '.jpg'])
            count_num = count_num + 1;
            kinp = input('Go on?[c]','s');
%             dsaveS = 255*ones(H,W,3);
%             dsave1 = dsave*rgb(1);
%             dsave1(lsave ~= 1) = 255;
%             dsave2 = dsave*rgb(2);
%             dsave2(lsave ~= 1) = 255;
%             dsave3 = dsave*rgb(3);
%             dsave3(lsave ~= 1) = 255;
%             dsaveS(:,:,1) = dsave1;dsaveS(:,:,2) = dsave2;dsaveS(:,:,3) = dsave3;
        end
        close(1)
    end
end