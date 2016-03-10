# small script to test how the coefficients of A change with respect
# to omega (or k in this case)

# We want to have a better PML, for that we need a PML that depends on
# the local wave-speed


# Clean version of the DDM ode

using PyPlot
using Devectorize


include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
#Defining Omega
h = 0.005
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
# the good modification for the quadrature
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# we will need to build an interpolant depending on the
# wave number

Aentriesk = entriesSparseA(k,X,Y,D[1], n ,m)[2]

Aentries2k = entriesSparseA(0.5*k,X,Y,D[2], n ,m)[2]

Aentries4k = entriesSparseA(0.25*k,X,Y,D[3], n ,m)[2]


println(Aentriesk[1][2])

println(Aentries2k[1][2])

println(Aentries4k[1][2])

#G = buildConvMatrix(k,X,Y,D0,h)


#the domain is decomposed in two subdomains

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.4).*(abs(y).<0.4);

Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

u_inc = exp(k*im*Y);

# bdyInd = setdiff(collect(1:N), volInd);

As = buildSparseA(k,X,Y,D0, n ,m);

AG = buildSparseAG(k,X,Y,D0, n ,m);

# need to check everything here :S

Mapproxsp = k^2*(AG*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;


S1 = Subdomain(As,AG,Mapproxsp,x,y, 1, round(Integer, m/3), 30, h, nu, k);
S2 = Subdomain(As,AG,Mapproxsp,x,y,  round(Integer, m/3)+1, round(Integer, 2*m/3), 30, h, nu, k);
S3 = Subdomain(As,AG,Mapproxsp,x,y,  round(Integer, 2*m/3)+1, m, 30, h, nu, k);

Minv = lufact(Mapproxsp);

rhs = zeros(Complex128,n,m)

RHS = reshape(rhs, n,m);
RHS[(end-1)/2,(end-1)/2 ] = n*m;
uGreen2 = Minv\RHS[:];

imshow(real(reshape(uGreen,n,m)))

rhs = RHS[:];

 rhsA2 = rhs[S2.indVolInt]
rhsLocal2 = zeros(Complex128,S2.n*S2.m)
rhsLocal2[S2.indVolIntLocal] = rhsA2

figure(2);
imshow(reshape(real(rhsLocal2),S2.n, S2.m))

uGreen2Local = solve(S2, rhsLocal2)

figure(3);
imshow(real(reshape(uGreen2Local,S2.n,S2.m)))


uGreenGlobal2 =  uGreen2[S2.indVolInt];
uGreenlocalInt = uGreen2Local[S2.indVolIntLocal];

segment = round(Integer, length(uGreenGlobal2)/m);

figure(4)
imshow(abs(reshape( uGreenGlobal2 - uGreenlocalInt, n, segment )/maximum(abs(uGreenGlobal2)) ) );
colorbar()


## for the last subdomains
rhs = zeros(Complex128,n,m)

RHS = reshape(rhs, n,m);
RHS[ceil(5*(end-1)/6),ceil(5*(end-1)/6) ] = n*m;
uGreen3 = Minv\RHS[:];

#figure(4)
#imshow(real(reshape(uGreen3,n,m)))


rhs = RHS[:];

 rhsA3 = rhs[S3.indVolInt]
rhsLocal3 = zeros(Complex128,S3.n*S3.m)
rhsLocal3[S3.indVolIntLocal] = rhsA3

figure(5);
imshow(reshape(real(rhsLocal3),S3.n, S3.m))

uGreen3Local = solve(S3, rhsLocal3)

figure(6);
imshow(real(reshape(uGreen3Local,S3.n,S3.m)))


uGreenGlobal3 =  uGreen3[S3.indVolInt];
uGreenlocalInt = uGreen3Local[S3.indVolIntLocal];

segment = round(Integer, length(uGreenGlobal3)/m);

figure(8)
imshow(abs(reshape( uGreenGlobal3 - uGreenlocalInt, n, segment )/maximum(abs(uGreenGlobal3)) ) );
colorbar()
