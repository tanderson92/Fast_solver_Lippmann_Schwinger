# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,
# in this file we add the possibility to use Sparse MKL for the matrix
# vector multiplication


# Clean version of the DDM ode

using PyPlot
using Devectorize
using IterativeSolvers
using Pardiso

include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
include("../src/preconditioner.jl")

#Defining Omega
h = 0.0025
k = (1/h)/2

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

nSubdomains = 16;
println("Number of Subdomains is ", nSubdomains)
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[2];

# the perturbation

nu(x,y)=  -(1.200*exp(-8000*((x-0.42).^2 + (y+0.42).^2))  +
            1.100*exp(-8000*((x+0.31).^2 + (y+0.30).^2))  +
            1.100*exp(-8000*((x-0.12).^2 + (y-0.32).^2))  +
            1.140*exp(-8000*((x+0.12).^2 + (y+0.32).^2))  +
            1.200*exp(-8000*((x-0.12).^2 + (y+0.22).^2))  +
            1.200*exp(-8000*((x-0.42).^2 + (y+0.42).^2))  +
            1.230*exp(-8000*((x-0.25).^2 + (y+0.00).^2))  +
            1.250*exp(-8000*((x+0.21).^2 + (y+0.21).^2))  +
            1.137*exp(-8000*((x+0.23).^2 + (y-0.30).^2))  +
            1.050*exp(-8000*((x-0.08).^2 + (y+0.23).^2))  +
            1.121*exp(-8000*((x-0.37).^2 + (y-0.29).^2))  +
            1.140*exp(-8000*((x-0.36).^2 + (y+0.10).^2))  +
            1.148*exp(-8000*((x+0.32).^2 + (y+0.35).^2))  +
            1.037*exp(-8000*((x-0.23).^2 + (y+0.30).^2))  +
            1.150*exp(-8000*((x+0.08).^2 + (y-0.23).^2))  +
            1.121*exp(-8000*((x+0.27).^2 + (y+0.29).^2))  +
            1.100*exp(-8000*((x+0.31).^2 + (y+0.30).^2))  +
            1.100*exp(-8000*((x-0.12).^2 + (y-0.32).^2))  +
            1.100*exp(-8000*((x+0.12).^2 + (y-0.32).^2))  +
            1.100*exp(-8000*((x-0.12).^2 + (y-0.22).^2))  +
            1.200*exp(-8000*((x-0.22).^2 + (y-0.22).^2))  +
            1.230*exp(-8000*((x-0.25).^2 + (y+0.00).^2))  +
            1.250*exp(-8000*((x-0.21).^2 + (y+0.21).^2))  +
            1.137*exp(-8000*((x-0.23).^2 + (y-0.30).^2))  +
            1.150*exp(-8000*((x+0.08).^2 + (y+0.23).^2))  +
            1.121*exp(-8000*((x+0.37).^2 + (y-0.29).^2))  +
            1.140*exp(-8000*((x+0.36).^2 + (y-0.10).^2))  +
            1.148*exp(-8000*((x+0.32).^2 + (y-0.35).^2))  +
            1.037*exp(-8000*((x+0.23).^2 + (y+0.30).^2))  +
            1.150*exp(-8000*((x-0.08).^2 + (y+0.23).^2))  +
            1.121*exp(-8000*((x-0.27).^2 + (y+0.29).^2))  +
            1.140*exp(-8000*((x-0.26).^2 + (y+0.10).^2))  +
            1.100*exp(-8000*((x+0.31).^2 + (y+0.30).^2))  +
            1.100*exp(-8000*((x-0.12).^2 + (y-0.32).^2))  +
            1.140*exp(-8000*((x+0.12).^2 + (y+0.32).^2))  +
            1.200*exp(-8000*((x-0.18).^2 + (y+0.22).^2))  +
            1.200*exp(-8000*((x-0.42).^2 + (y+0.42).^2))  +
            1.230*exp(-8000*((x-0.25).^2 + (y+0.00).^2))  +
            1.250*exp(-8000*((x+0.21).^2 + (y+0.21).^2))  +
            1.137*exp(-8000*((x+0.27).^2 + (y-0.30).^2))  +
            1.150*exp(-8000*((x-0.08).^2 + (y+0.23).^2))  +
            1.121*exp(-8000*((x-0.37).^2 + (y-0.29).^2))  +
            1.140*exp(-8000*((x-0.36).^2 + (y+0.10).^2))  +
            1.148*exp(-8000*((x+0.32).^2 + (y+0.35).^2))  +
            1.037*exp(-8000*((x-0.01).^2 + (y+0.11).^2))  +
            1.150*exp(-8000*((x+0.27).^2 + (y-0.12).^2))  +
            1.221*exp(-8000*((x+0.25).^2 + (y+0.23).^2))  +
            1.100*exp(-8000*((x+0.13).^2 + (y+0.31).^2))  +
            1.100*exp(-8000*((x-0.13).^2 + (y-0.05).^2))  +
            1.100*exp(-8000*((x+0.17).^2 + (y-0.04).^2))  +
            1.100*exp(-8000*((x-0.32).^2 + (y-0.19).^2))  +
            1.200*exp(-8000*((x-0.08).^2 + (y-0.46).^2))  +
            1.230*exp(-8000*((x-0.05).^2 + (y+0.04).^2))  +
            1.250*exp(-8000*((x-0.10).^2 + (y+0.20).^2))  +
            1.137*exp(-8000*((x-0.30).^2 + (y-0.40).^2))  +
            1.150*exp(-8000*((x+0.19).^2 + (y+0.47).^2))  +
            1.221*exp(-8000*((x+0.32).^2 + (y-0.00).^2))  +
            1.140*exp(-8000*((x+0.33).^2 + (y-0.15).^2))  +
            1.248*exp(-8000*((x+0.12).^2 + (y-0.09).^2))  +
            1.137*exp(-8000*((x+0.29).^2 + (y+0.25).^2))  +
            1.150*exp(-8000*((x-0.27).^2 + (y+0.40).^2))  +
            1.021*exp(-8000*((x-0.25).^2 + (y+0.06).^2))  +
            1.140*exp(-8000*((x-0.33).^2 + (y+0.13).^2))  +
            1.048*exp(-8000*((x-0.09).^2 + (y-0.03).^2))).*(abs(x).<0.48).*(abs(y).<0.48);

nuT(x,y)= nu(y,x)

Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);
figure(1); clf();
imshow(real(reshape( 1./(1- nu(X,Y)),n,m)), extent=[x[1], x[end], y[1], y[end]]);cb =  colorbar();

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
T = speye(N);

index = 1:N;
index = (reshape(index, n,m).')[:];

T = T[index,:];

MapproxspT =  T*Mapproxsp*T.';
AGT = T*AG*T.';
AsT = T*As*T.';
SubArray1 = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii],idxn[ii], npml, h, nu, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];
SubArray2 = [ Subdomain(AsT,AGT,MapproxspT,x,y, idx1[ii],idxn[ii], npml, h, nuT, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];


# this step is a hack to avoid building new vectors in int32 every time
for ii = 1:nSubdomains
	convert64_32!(SubArray1[ii])
      convert64_32!(SubArray2[ii])
end

tic();
for ii=1:nSubdomains
	factorize!(SubArray1[ii])
      factorize!(SubArray2[ii])
end
println("Time for the factorization ", toc())

# # Testing the Polarized traces preconditioner
# GSPrecond1 = GSPreconditioner(SubArray1);
# GSPrecond2 = GSPreconditioner(SubArray2);


# GSdouble = doublePreconditioner(SubArray1,SubArray2,Mapproxsp);


# u_inc = exp(k*im*X);
# rhs = -(fastconv*u_inc - u_inc);

# u = zeros(Complex128,N);

# info = gmres!(u, Mapproxsp, As*rhs, GSdouble);

# # # testing the preconditioner for the sparsified system
# # u = zeros(Complex128,N);
# # @time info =  gmres!(u, Mapproxsp, As*rhs,GSPrecond , restart = 20)
# # println("number of iterations for inner solver using a Gauss-Seidel preconditioner is ", countnz(info[2].residuals[:]))

# # instanciating the new preconditioner
# Precond1 = Preconditioner(As,Mapproxsp,SubArray1);
# Precond2 = Preconditioner(AsT,MapproxspT,SubArray2);
# # PrecondMKL = Preconditioner(As,Mapproxsp,SubArray, mkl_sparseBlas = true)

doublePrecond = doublePreconditioner(As,Mapproxsp,SubArray1,SubArray2);

# building the RHS from the incident field
# theta = rand(1)[1]*2*pi
# u_inc = exp(k*im*(X*cos(theta) + Y*sin(theta)));
u_inc = exp(k*im*Y);
rhs = -(fastconv*u_inc - u_inc);

u = zeros(Complex128,N);
@time info =  gmres!(u, fastconv, rhs, doublePrecond)

# u = zeros(Complex128,N);
# @time info2 =  gmres!(u, fastconv, rhs, PrecondMKL)

println(info[2].residuals[:])
println("Number of iterations to convergence is ", countnz(info[2].residuals[:]))

figure(2); clf();
imshow(real(reshape(u+u_inc,n,m)), extent=[x[1], x[end], y[1], y[end]])

tick_params(fontsize=22)