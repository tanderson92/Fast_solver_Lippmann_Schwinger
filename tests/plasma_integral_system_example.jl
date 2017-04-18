# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,
# in this file we add the possibility to use Sparse MKL for the matrix
# vector multiplication


# Clean version of the DDM ode

using PyPlot
using Devectorize
using IterativeSolvers
#using Pardiso

include("../src/SparsifyingMatrix2D.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
include("../src/preconditioner.jl")
include("../src/integral_preconditioner.jl")

#Defining Omega
h = 0.0025
k = (1/h)

# setting the correct number of threads for FFTW and
# BLAS, this can be further
FFTW.set_num_threads(16)
blas_set_num_threads(16)

println("Frequency is ", k/(2*pi))
println("Number of discretization points is ", 1/h)

# size of box
a  = 1
x = -a/2:h:a/2
y = -a/2:h:a/2
n = length(x)
m = length(y)
N = n*m
X = repmat(x, 1, m)[:]
Y = repmat(y', n,1)[:]

nSubdomains = 8;
println("Number of Subdomains is ", nSubdomains)
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# the perturbation
C = 0.4987
phi(x,y) = 1 - (x-0.05*(1-x.^2)).^2 - C*((1 + 0.3x).^2).*y.^2

aa = [0.45, 0.196, 0.51, 0.195, 0.63 ];
xI = [0.4 , 0.54 , -0.14, -0.5, 0.18];
yI = [0   , -0.28, 0.70, -0.01, 0.8];

g(x,y) = aa[1]*exp(-((x-xI[1]).^2 + (y-yI[1]).^2)/0.01 ) +
         aa[2]*exp(-((x-xI[2]).^2 + (y-yI[2]).^2)/0.01 ) +
         aa[3]*exp(-((x-xI[3]).^2 + (y-yI[3]).^2)/0.01 ) +
         aa[4]*exp(-((x-xI[4]).^2 + (y-yI[4]).^2)/0.01 ) +
         aa[5]*exp(-((x-xI[5]).^2 + (y-yI[5]).^2)/0.01 ) ;

nu1(x,y)=  (phi(x,y).>0.05).*(-1.5*(phi(x,y)-0.05) - g(x,y).*cos(0.9*y))  ;

nu(x,y) = -nu1(3*x,3*y); # sign is important the convention with respect to Leslie paper is different


Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);
figure(1); clf();
imshow(real(reshape( (1- nu(X,Y)),n,m)), extent=[x[1], x[end], y[1], y[end]]);cb =  colorbar();

println("Building the A sparse")
@time As = buildSparseA(k,X,Y,D0, n ,m);

println("Building A*G in sparse format")
@time AG = buildSparseAG(k,X,Y,D0, n ,m);

# need to check everything here :S

Mapproxsp = k^2*(AG*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

# number of interior points for each subdomain
SubDomLimits = round(Integer, floor(linspace(1,m+1,nSubdomains+1)))
# index in y. of the first row of each subdomain
idx1 = SubDomLimits[1:end-1]
# index in y of the last row of each subdomains
idxn = SubDomLimits[2:end]-1

#npml = round(Integer, ((m-1)/nSubdomains)/2)
npml = 10

#SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], npml, h, nu, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];
SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], npml, h, nu, k) for ii = 1:nSubdomains];


# # this step is a hack to avoid building new vectors in int32 every time
# for ii = 1:nSubdomains
# 	convert64_32!(SubArray[ii])
# end

tic();
for ii=1:nSubdomains
	factorize!(SubArray[ii])
end
println("Time for the factorization ", toc())

Precond = PolarizedTracesPreconditioner(As, SubArray, nIt =2);

# building the RHS from the incident field
# theta = rand(1)[1]*2*pi
# u_inc = exp(k*im*(X*cos(theta) + Y*sin(theta)));
u_inc = exp(k*im*X);
rhs = -(fastconv*u_inc - u_inc);

u = zeros(Complex128,N);
@time info =  gmres!(u, fastconv, rhs, Precond)
println("Number of iterations to convergence is ", countnz(info[2].residuals[:]))


u_inc2 = exp(k*im*Y);
rhs = -(fastconv*u_inc2 - u_inc2);

u2 = zeros(Complex128,N);
@time info =  gmres!(u2, fastconv, rhs, Precond)

println("Number of iterations to convergence is ", countnz(info[2].residuals[:]))

figure(2); clf();
imshow(real(reshape(u+u_inc,n,m).'), extent=[x[1], x[end], y[1], y[end]])

figure(3); clf();
imshow(real(reshape(u2+u_inc2,n,m).'), extent=[x[1], x[end], y[1], y[end]])
