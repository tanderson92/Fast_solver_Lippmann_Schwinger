# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,
# we modify everything in order to obtan the correct boundary conditions

using PyPlot
using Devectorize


include("FastConvolution.jl")
include("quadratures.jl")
#Defining Omega
h = 0.02
k = 1/h

# size of box
a  = 1
x = -a/2:h:a/2
y = -a/2:h:a/2
n = length(x)
m = length(y)
N = n*m
X = repmat(x, 1, m)[:]
Y = repmat(y', n,1)[:]
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[1];



x1 = -a/2:h:a/2
y1 = y[1:round(Integer, ceil(3*end/4))]
n1 = length(x1)
m1 = length(y1)
N1 = n1*m1
X1 = repmat(x1, 1, m1)[:]
Y1 = repmat(y1', n1,1)[:]

x2 = -a/2:h:a/2
y2 = y[round(Integer, ceil(end/4)):end]
n2 = length(x2)
m2 = length(y2)
N2 = n2*m2
X2 = repmat(x2, 1, m2)[:]
Y2 = repmat(y2', n2,1)[:]


#the domain is decomposed in two subdomains

filter1(x,y) = 0*x + (y.<=2*h) + (y.>2*h).*(y.<y1[end-2]).*exp(-100*(y-2*h).^2)
filter2(x,y) = 0*x + (y.>=-h) + (y.<-h).*(y.>y2[3]).*exp(-100*(y+h).^2)

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.4).*(abs(y).<0.4);

nu1(x,y) = filter1(x,y).*nu(x,y);
nu2(x,y) = filter2(x,y).*nu(x,y);



# building the convolution matrices for the subdomains

G = buildConvMatrix(k,X,Y,D0,h)
G1 = buildConvMatrix(k,X1,Y1,D0,h);
G2 = buildConvMatrix(k,X2,Y2,D0,h);

As = buildSparseA(k,X,Y,D0, n ,m);
As1 = buildSparseA(k,X1,Y1,D0, n1 ,m1);
As2 = buildSparseA(k,X2,Y2,D0, n2 ,m2);

Mapprox = zeros(Complex128, N,N);
Mapprox = As*G;
Mapproxtemp = sparse(Mapprox.*(abs(As).>0));

Mapproxsp = k^2*(Mapproxtemp*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;


Mapprox1 = zeros(Complex128, N,N);
Mapprox1 = As1*G1;
Mapprox1temp = sparse(Mapprox1.*(abs(As1).>0));

Mapprox1sp = k^2*(Mapprox1temp*spdiagm(nu1(X1,Y1)));
Mapprox1sp = As1 + Mapprox1sp;

Mapprox2 = zeros(Complex128, N,N);
Mapprox2 = As2*G2;
Mapprox2temp = sparse(Mapprox2.*(abs(As2).>0));

Mapprox2sp = k^2*(Mapprox2temp*spdiagm(nu2(X2,Y2)));
Mapprox2sp = As2 + Mapprox2sp;


Minv  = lufact(Mapproxsp);
M1inv = lufact(Mapprox1sp);
M2inv = lufact(Mapprox2sp);

# defining some indices

indVolInt1 =  find(Y.< h)
indVol1   =  find(Y.<= y1[end])
indVolIntLocal1 =  find(Y1.< h)

indVolInt2 =  find(Y.> 0)
indVol2   =  find(Y.>= y2[1])
indVolIntLocal2 =  find(Y2.> 0)

ind1_N = find(Y1.==0);
ind1_N1 = find(Y1.==h);

ind2_0 = find(Y2.==0);
ind2_1 = find(Y2.==h);


# building a dirac delta
dirac = zeros(Complex128, N)
dirac[n*(n-1)/2 + (n+1)/2] = 1;

diracA = As*dirac;

dirac1 = zeros(Complex128,N1)

dirac1[indVolIntLocal1] = diracA[indVolIntLocal1]

diracA1 = zeros(Complex128,N1)
diracA1[n*(n+1)/2 + (n+1)/2] = 1;

diracA1 = As1*diracA1;

Gfunc = Minv\diracA;

Gfunc1 = M1inv\dirac1;
GfuncA1 = M1inv\diracA1;

figure(1);
imshow(real(reshape(Gfunc,n,m)))

figure(2);
imshow(real(reshape(Gfunc1,n1,m1)))

figure(3);
imshow(real(reshape(GfuncA1,n1,m1)))


# trying to build the boundary conditions

indVolInt1 =  find(Y.< h)
indVol1   =  find(Y.<= y1[end])
indVolIntLocal1 =  find(Y1.< h)

Mapproxsp1 =  (As + k^2*(Mapproxtemp*spdiagm(nu1(X,Y))))[indVol1,indVol1];

indVolInt2 =  find(Y.> 0)
indVol2   =  find(Y.>= y2[1])
indVolIntLocal2 =  find(Y2.> 0)
Mapproxsp2 =  (As + k^2*(Mapproxtemp*spdiagm(nu2(X,Y))))[indVol2,indVol2];

Mapproxsp1[end-2*n+1:end, end-2*n+1:end]=Mapproxsp[end-2*n+1:end, end-2*n+1:end];

M1spinv = lufact(Mapproxsp1);
M2spinv = lufact(Mapproxsp1);


Gfunc1sp = M1spinv\dirac1;
GfuncA1sp = M1spinv\diracA1;

figure(4);
imshow(real(reshape(Gfunc1sp,n1,m1)))

figure(5);
imshow(real(reshape(GfuncA1sp,n1,m1)))

#########

Mapproxsp1mod = copy(Mapproxsp1)




