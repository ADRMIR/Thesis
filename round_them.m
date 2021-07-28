%This function chooses a rounding threshold

function y_hat =round_them(y,thresh)

      for k=1:length(y)        %Goes through the output vector to approxiate outliers into the supposed interval more than one equals unity, less than zero equals nullity
            if y(k)>thresh
                y_hat(k)=1;
            elseif  y(k)<=thresh
                y_hat(k)=0;
            end
        end