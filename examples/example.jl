# small scrip to compute the solution of Lippmann-Schwinger equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.

using PyCall
pygui(:qt5)
using PyPlot
pygui(true)
ion()
using IterativeSolvers
using SpecialFunctions
using SparseArrays
using Distributed
using SharedArrays
using LinearAlgebra
using FFTW
using Pardiso
using LinearMaps


include("../src/SparsifyingMatrix2D.jl")
include("../src/preconditioner.jl")


# setting the number of threads for the FFT and BLAS
# libraries (please set them to match the number of
# physical cores in your system)
FFTW.set_num_threads(16);
BLAS.set_num_threads(16);


#Defining Omega
#h = 0.000625
#h = 0.00125
h = 0.0025
#h = 0.005
#h = 0.01
#k = 1/(h)
k = 3π

# size of box
a = 2
x = collect(-a/2:h:a/2)
y = collect(-a/2:h:a/2)
(n,m) = length(x), length(y)
N = n*m
X = repeat(x, 1, m)[:]
Y = repeat(y', n,1)[:]
# we solve \triangle u + k^2(1 + nu(x))u = 0

# We use the modified quadrature in Duan and Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

# Defining the smooth perturbation of the slowness
nu(x,y) = @. 0.7*exp(-40*(x.^2 + y.^2)).*(abs(x).<0.97).*(abs(y).<0.97);
#nu(x, y) = @. 1 + 0.0*x + 0.0*y

## You can choose between Duan Rohklin trapezoidal quadrature
#fastconv = buildFastConvolution(x,y,h,k,nu)
fastconv = buildFastConvolution(x, y, h, k, nu, quadRule = "trapezoidal");
## WARNING: You *must* use rhs = -(fastconv*u_inc - u_inc) below if you use Duan-Rokhlin quadrature

# or Greengard Vico Quadrature (this is not optimized and is 2-3 times slower)
#fastconv = buildFastConvolution(x, y, h, k, nu, quadRule = "Greengard_Vico");

# wrapper for linear maps
function apply_conv!(x)
    return fastconvolution(fastconv,x) 
end

convolution_map = LinearMap(apply_conv!, size(fastconv)[1][1]; issymmetric=false, ismutating=false)

# assembling the sparsifiying preconditioner
@time As = buildSparseA(k, X, Y, D0, n, m);

# assembling As*( I + k^2G*nu)
@time Mapproxsp = As + k^2*(buildSparseAG(k, X, Y, D0, n ,m)*spdiagm(nu(X,Y)));

# defining the preconditioner
# # We use UMFPACK by default
precond = SparsifyingPreconditioner(Mapproxsp, As);
# We use MKLPARDISO
#precond = SparsifyingPreconditioner(Mapproxsp, As; solverType="MKLPARDISO")

# building the RHS from the incident field
u_inc = exp.(k*im*Y);
rhs = -k^2*FFTconvolution(fastconv, nu(X,Y).*u_inc) ;
#rhs = -(fastconv*u_inc - u_inc);

#rhs = u_inc;

# allocating the solution
u = zeros(Complex{Float64},N);

# solving the system using GMRES
@time info =  gmres!(u, fastconv, rhs, Pl=precond, log = true, verbose = true)
println(info[2].data[:resnorm])
println(size(info[2].data[:resnorm]))


#u = zeros(Complex{Float64},N);
#@time info =  gmres!(u, fastconv, rhs, log = true)
#println(info[2].data[:resnorm])
#println(size(info[2].data[:resnorm]))

# plotting the solution
figure(1)
clf()
#imshow(real(reshape(u+u_inc,n,m)))
utot_reshape = reshape(u+u_inc, n, m)
utot_plt = utot_reshape[2:end,2:end]
u_reshape = reshape(u, n, m)
u_plt = u_reshape[2:end,2:end]
Xmat = reshape(X, n, m)
Ymat = reshape(Y, n, m)
# Visualize the scattered wave solution
pcolormesh(Xmat, Ymat, real(utot_plt))

#indx = round(Int, -(-1 - -0.10278700444897992)/h) + 1
#indy = round(Int, -(-1 - 0.3052396731497715)/h) + 1
#
#indx2 = round(Int, -(-1 - 0.39414344533005496)/h) + 1
#indy2 = round(Int, -(-1 - -0.3189957194709198)/h) + 1
#
#using FastGaussQuadrature
#σ = 1/9
#xx, ww = gausslegendre(100)
#function Grho_anly(x, y)
#    r = norm((x, y))
#    a1 = 0; b1 = r
#    a2 = r; b2 = 1.2
#    f1 = (y) -> besselj0(k*y)*exp(-y^2/(2*σ^2)) * y
#    f2 = (y) -> hankelh1(0, k*y)*exp(-y^2/(2*σ^2)) * y
#    I1 = hankelh1(0, k*r) / (4*σ^2) * dot((b1-a1)/2*ww, f1.((b1-a1)/2*xx .+ (a1 + b1)/2))
#    I2 = besselj0(k*r) / (4 * σ^2) * dot((b2-a2)/2*ww, f2.((b2-a2)/2*xx .+ (a2+b2)/2))
#    I1 + I2
#end
#
#
#ρvec = 1/(σ^2 * 2π) * exp.(-(X.^2 + Y.^2)/(2 * σ^2)) .+ 0im
#Grho_eval = reshape(fastconv * ρvec, n, m)
#ρ = 1/(σ^2 * 2π) * exp.(-(Xmat.^2 + Ymat.^2)/(2 * σ^2))
#
#pt = (Xmat[indx, indy], Ymat[indx, indy])
#Grho_anly_eval = Grho_anly.(Xmat, Ymat)