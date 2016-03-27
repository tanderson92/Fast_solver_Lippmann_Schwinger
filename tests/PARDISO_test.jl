# This file implmentes the Sparsifying preconditioner using MKL Pardiso and as a
# the built-in UMFPACK as multifrontal solvers


using PyPlot
using Devectorize
using IterativeSolvers
using Pardiso

include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")

#Defining Omega
h = 0.01
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

nSubdomains = 4;
println("Number of Subdomains is ", nSubdomains)
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the quadrature modification by Duan Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# defining the perturbation
# in this case the plasma profile was given by Antoine Cerfon
nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);

Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

# definin the fast convolution type (which uses a Toeplitz embedding and
# the FFT)
fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

println("Building the A sparse")
As = buildSparseA(k,X,Y,D0, n ,m);

println("Building A*G in sparse format")
AG = buildSparseAG(k,X,Y,D0, n ,m);


Mapproxsp = k^2*(AG*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

# in this case we use the built-in LU factorization, which relies on UMFPACK
Minv = lufact(Mapproxsp)

precond(x) = Minv\(As*(fastconv*x));

# building the RHS from the incident field
u_inc = exp(k*im*Y);
# rhs = omega^2 G*nu*u_inc
rhs = -(fastconv*u_inc - u_inc);

# allocating the solution
u = zeros(Complex128,N);
# solving the Lippmann Schwinger equation using gmres and UMFPACK
@time info =  gmres!(u, precond, Minv\(As*rhs))
println(info[2].residuals[:])

# defining the mkl solver (this needs to be encapsulated)

solverMKL = MKLPardisoSolver();

set_nprocs(solverMKL, 16)
set_mtype(solverMKL, 3)
set_iparm(solverMKL,12, 2)


#xx0 = Minv\bb
bb = (As*rhs)
x0 = zeros(bb)
set_phase(solverMKL, 12)
pardiso(solverMKL, x0, Mapproxsp, bb)

set_iparm(solverMKL,12, 2)
set_phase(solverMKL, 33)
pardiso(solverMKL, x0, Mapproxsp, bb)

# encapculating the Sparsifying preconditioner
function precondMKL(b)
    x = zeros(b)
    pardiso(solverMKL, x, Mapproxsp, (As*(fastconv*b)))
    return x
end

u = zeros(Complex128,N);
# solving the Lippmann Schwinger equation using gmres and PARDISO
@time info =  gmres!(u, precondMKL, x0)
println(info[2].residuals[:])



