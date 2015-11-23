# small scrip to compute the solution of Lipman Schinwer equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.


using PyPlot
using Devectorize


include("FastConvolution.jl")
include("quadratures.jl")
#Defining Omega
h = 0.0005
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


nu(x,y) = -0.3*exp(-40*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);
#nu(x,y) = ((abs(x)+abs(y)).<0.2)

xe = collect((x[1]-(n-1)*h):h:(x[end]+(n-1)*h));
ye = collect((y[1]-(m-1)*h):h:(y[end]+(m-1)*h));

Xe = repmat(xe, 1, 3*m-2);
Ye = repmat(ye', 3*n-2,1);

R = sqrt(Xe.^2 + Ye.^2);
# to avoid evaluating at the singularity
indMiddle = round(Integer, m-1 + (m+1)/2)
R[indMiddle,indMiddle] =1;
Ge = 1im/4*hankelh1(0, k*R)*h^2;
Ge[indMiddle,indMiddle] = 1im/4* D0*h^2;

GFFT = fft(Ge);
fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

u_inc = exp(k*im*X);

# bdyInd = setdiff(collect(1:N), volInd);

As = buildSparseA(k,X,Y,D0, n ,m);

Mapproxsp = buildSparseAG(k,X,Y,D0, n ,m);
Mapproxsp = k^2*(Mapproxsp*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

Minv = lufact(Mapproxsp);

using IterativeSolvers


precond(x) = (Minv\(As*(fastconv*x)));

# building the RHS from the incident field
u_inc = exp(k*im*X);
rhs = -(fastconv*u_inc - u_inc);
#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


u= zeros(Complex128,N);
@time info =  gmres!(u, precond, Minv\(As*rhs))
info[2].residuals[:]
