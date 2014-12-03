clear all;
buildmexexample
mexexample('init');

a = single(complex(2, 3));
X = complex(ones(4000, 4000, 'single')*5., ones(4000, 4000, 'single') * 6.);
Y = complex(ones(4000, 4000, 'single')*7., ones(4000, 4000, 'single') * 8.);

Z1 = mexexample('axpy', a, X, Y);
B = mexexample('compute', X);

Z2 = a*X+Y; 
if isequal(Z1, Z2)
    disp('good!');
else
    disp('ERROR!')
end
tic;
for i=1:100
   Z2 = a*X+Y; 
end
toc;