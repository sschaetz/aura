function performancetest(T,I,N)
%%performance test
%specify parameters according to 'Outsourcing of Operators to GPU.lyx'

if nargin < 1
T=100;
end 
if nargin < 2
I=10;
end
if nargin <3
N=256;
end

y=rand(N^2*3,1);
myDF=rand(N,N,T,3);
P.CoilProf=rand(N,N,I);
P.dimos=[N,N];
P.frames=T;
P.nc=I;
P.U_Mask=rand(N,N,T);
P.FOVmask=rand(N,N);

fct = @(y) real(opd_y(op_y(y,P,myDF),P,myDF));

%take best out of 3 tic toc
t=+inf;
for i = 1:3
    tic,fct(y);
    t=min(t,toc);
end
fprintf('Best out of 3 was %2.2f sec\n',t);

% backward operator
function opd_y = opd_y(yl,P,DF)
    %last argument is Jacobi matrix
    if ~isfield(P,'CoilProf')
        C=1;
    else
        C=P.CoilProf;
    end
    yl    = reshape(yl,[P.dimos,P.frames,P.nc]);
    yl    = bsxfun(@times,yl,P.U_Mask);
    yl    = nifft2(yl);%uses ifft2 with 'symmetric'
%     yl    = real(yl);
    yl    = bsxfun(@times,yl,P.FOVmask);
    yl    = bsxfun(@times,yl,reshape(conj(C),[P.dimos 1 P.nc]));
    yl    = sum(yl,4);
    opd_y = bsxfun(@times,DF,yl);
    opd_y = sum(opd_y,3);
    opd_y = opd_y(:);
end

%forward operator
function op_y = op_y(y,P,DF)
    %last argument is Jacobi matrix
    if ~isfield(P,'CoilProf')
        C=1;
    else
        C=P.CoilProf;
    end
    y     = reshape(y,[P.dimos,1,3]);
%     op_y  = bsxfun(@times,DF,reshape(P.sc,[P.dimos,1,3]));
    op_y  = bsxfun(@times,DF,P.FOVmask);
    op_y  = bsxfun(@times,op_y,y);
    op_y  = sum(op_y,4);
    C     = reshape(C,[P.dimos 1 P.nc]);
    op_y  = bsxfun(@times,op_y,C);
    op_y  = nfft2(op_y);
    op_y  = bsxfun(@times,op_y,P.U_Mask);
    op_y  = op_y(:);
end
    
end
