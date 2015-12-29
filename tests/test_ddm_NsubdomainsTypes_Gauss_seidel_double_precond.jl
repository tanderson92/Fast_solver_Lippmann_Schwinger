# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,


# Clean version of the DDM ode

using PyPlot
using Devectorize
using IterativeSolvers
#using Pardiso

include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
include("../src/preconditioner.jl")
#Defining Omega
h = 0.0025
k = 1/h

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

nSubdomains = 12;
println("Number of Subdomains is ", nSubdomains)
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

#G = buildConvMatrix(k,X,Y,D0,h)


#the domain is decomposed in two subdomains

nu(x,y) = ( 0.2*exp(-20*(x.^2 + y.^2))  + 0.25*exp(-200*((x - 0.1).^2 + (y+0.1).^2)) +
            0.2*exp(-2000*((x+0.1).^2 + y.^2))  + 0.25*exp(-1000*((x -0.1).^2 + (y+0.1).^2)) +
            0.1*exp(-200*((x+0.3).^2 + (y+0.3).^2))  + 0.1*exp(-1000*((x-0.1).^2 + (y+0.2).^2)) +
            0.1*exp(-200*((x-0.12).^2 + (y-0.32).^2))  + 0.1*exp(-1000*((x+0.14).^2 + (y+0.24).^2)) +
            0.1*exp(-200*((x+0.12).^2 + (y+0.32).^2))  + 0.1*exp(-1000*((x-0.14).^2 + (y-0.24).^2)) +
            0.1*exp(-200*((x-0.12).^2 + (y+0.22).^2))  + 0.1*exp(-1000*((x+0.34).^2 + (y+0.4).^2)) +
            0.2*exp(-1000*((x-0.42).^2 + 2*(y+0.42).^2)) + 0.1*exp(-1000*((x+0.14).^2 + (y+0.04).^2)) +
            0.3*exp(-2000*((x-0.25).^2 + 3*(y).^2))  + 0.2*exp(-1000*((x-0.34).^2 + (y-0.4).^2)) +
            0.25*exp(-200*((x+0.2).^2 + (y+0.2).^2))  + 0.2*exp(-1000*(2*(x-0.1).^2 + (y+0.2).^2)) ).*(abs(x).<0.48).*(abs(y).<0.48);

Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

u_inc = exp(k*im*Y);

imshow(real(reshape( 1- nu(X,Y),n,m)))
# bdyInd = setdiff(collect(1:N), volInd);

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

npml = round(Integer, (m-1)/nSubdomains)


tic();
#SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], 20, h, nu, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];
SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], npml, h, nu, k) for ii = 1:nSubdomains];
println("Time for the factorization ", toc())

# compute the right hand side
rhs = -(fastconv*u_inc - u_inc);


# Testing the Polarized traces preconditioner
# GSPrecond = GSPreconditioner(SubArray)

# # testing the preconditioner for the sparsified system
# u = zeros(Complex128,N);
# @time info =  gmres!(u, Mapproxsp, As*rhs,GSPrecond , restart = 20)
# println("number of iterations for inner solver using a Gauss-Seidel preconditioner is ", countnz(info[2].residuals[:]))

# instanciating the new preconditioner
Precond = Preconditioner(As,Mapproxsp,SubArray)

# building the RHS from the incident field
u_inc = exp(k*im*Y);
rhs = -(fastconv*u_inc - u_inc);
#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


 u = zeros(Complex128,N);
 @time info =  gmres!(u, fastconv, rhs, Precond)
println(info[2].residuals[:])
println("Number of iterations to convergence is ", countnz(info[2].residuals[:]))


imshow(real(reshape(u+u_inc,n,m)))
