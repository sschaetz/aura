%%
clear all;
% prepare some data

R = single(complex(rand(256,256,300), rand(256,256,300)));

Rm = ifft2(fft2(R));
tic;
Rm = ifft2(fft2(R));
toc;


Rg = gpufft(R);
tic;
Rg = gpufft(R);
toc;

%%
%tic;
%Rgm = gpuArray(R);
%Rgmf = gather(ifft2(fft2(Rgm)));
%toc;

