function [FinalResult2] = CWAN(im1,netCWANL,netCWANAB)

    imgF1   = double(rgb2gray(im1));
    imgF2   = double(rgb2lab(im1));
    imgF2 = imgF2(:,:,2:3);
    res=run_CWAN_Lightness(netCWANL,single(imgF1));
    result1 = im2single(gather(res(end).x));
    res=run_CWAN_AB_color(netCWANAB,single(imgF2));
    result2 = gather(res(end).x);
    result1 = (result1);
    result1 = result1 * 100 / 255;
    result1(:,:,2:3) = result2(:,:,1:2);
    FinalResult2 = lab2rgb(double(result1));    
    FinalResult2 = im2uint8(FinalResult2);
end

