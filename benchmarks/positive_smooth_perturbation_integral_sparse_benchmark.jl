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
include("../src/integral_preconditioner.jl")

#Defining Omega
H = [ 0.005, 0.0025, 0.002, 0.00125, 0.001, 0.0008, 0.0007142857142857143, 0.000625, 0.0005]
Subs = [  4,  8, 10, 16, 20, 25, 28, 32, 40]
NPML = [ 15, 15, 15, 15, 15, 15, 15, 15, 15]


for ll = 1:length(H)

    h = H[ll]

    nSubdomains = Subs[ll];

    npml = NPML[ll]
    
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
    
    
    println("Number of Subdomains is ", nSubdomains)
    # we solve \triangle u + k^2(1 + nu(x))u = 0
    # in particular we compute the scattering problem
    
    
    # we extract from a "tabulated" dictionary
    # the good modification for the quadrature modification
    (ppw,D) = referenceValsTrapRule();
    D0 = D[1];
    
    
    
    nu(x,y)=    -( 0.200*exp(-20*(x.^2 + y.^2))  + 0.25*exp(-200*((x - 0.1).^2 + (y+0.1).^2)) +
                0.200*exp(-2000*((x+0.11).^2 +   y.^2))         + 0.250*exp(-1000*(  (x-0.112).^2 + (y+0.1).^2)) +
                0.100*exp(-200* ((x+0.31).^2 +   (y+0.30).^2))  + 0.120*exp(-1000*(  (x-0.128).^2 + (y+0.2).^2)) +
                0.100*exp(-200* ((x-0.12).^2 +   (y-0.32).^2))  + 0.110*exp(-1000*(  (x+0.152).^2 + (y+0.24).^2)) +
                0.100*exp(-200* ((x+0.12).^2 +   (y+0.32).^2))  + 0.100*exp(-1000*(  (x-0.140).^2 + (y-0.24).^2)) +
                0.100*exp(-200* ((x-0.12).^2 +   (y+0.22).^2))  + 0.104*exp(-1000*(  (x+0.340).^2 + (y+0.35).^2)) +
                0.200*exp(-1000*((x-0.42).^2 + 2*(y+0.42).^2))  + 0.101*exp(-1000*(  (x+0.140).^2 + (y+0.04).^2)) +
                0.230*exp(-2000*((x-0.25).^2 + 3*(y+0.00).^2))  + 0.203*exp(-1000*(  (x-0.340).^2 + (y-0.31).^2)) +
                0.250*exp(-200* ((x+0.21).^2 +   (y+0.21).^2))  + 0.205*exp(-1000*(2*(x-0.101).^2 + (y+0.2).^2)) +
                0.037*exp(-200*( (x+0.23).^2 +   (y-0.30).^2))  + 0.208*exp(-1000*(  (x-0.274).^2 + (y-0.24).^2)) +
                0.050*exp(-200*( (x-0.08).^2 +   (y+0.23).^2))  + 0.233*exp(-100* (  (x-0.204).^2 + (y-0.24).^2)) +
                0.021*exp(-1000*((x-0.37).^2 + 2*(y-0.29).^2))  + 0.299*exp(-1000*(  (x-0.124).^2 + (y+0.04).^2)) +
                0.140*exp(-2000*((x-0.36).^2 + 3*(y+0.10).^2))  + 0.105*exp(-100 *(  (x-0.234).^2 + (y+0.14).^2)) +
                0.048*exp(-200*( (x+0.32).^2 +   (y+0.35).^2))  + 0.258*exp(-1000*(2*(x-0.051).^2 + (y-0.2).^2)) +
                0.037*exp(-200*( (x-0.23).^2 +   (y+0.30).^2))  + 0.208*exp(-1000*(  (x+0.274).^2 + (y+0.24).^2)) +
                0.050*exp(-200*( (x+0.08).^2 +   (y-0.23).^2))  + 0.233*exp(-100* (  (x+0.204).^2 + (y+0.24).^2)) +
                0.021*exp(-1000*((x+0.27).^2 + 2*(y+0.29).^2))  + 0.199*exp(-1000*(  (x+0.124).^2 + (y-0.04).^2)) +
                0.140*exp(-2000*((x+0.26).^2 + 3*(y-0.10).^2))  + 0.105*exp(-100* (  (x+0.234).^2 + (y-0.34).^2)) +
                0.048*exp(-200*( (x-0.20).^2 +   (y-0.39).^2))  + 0.258*exp(-1000*(2*(x+0.051).^2 + (y+0.2).^2)) ).*(abs(x).<0.48).*(abs(y).<0.48);
    
    
    
    nuT(x,y)= nu(y,x)
    
    Ge = buildGConv(x,y,h,n,m,D0,k);
    GFFT = fft(Ge);
    
    fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);
    
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
    T = speye(N);
    
    index = 1:N;
    index = (reshape(index, n,m).')[:];
    
    T = T[index,:];
    
    MapproxspT =  T*Mapproxsp*T.';
    AGT = T*AG*T.';
    AsT = T*As*T.';
    SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii],idxn[ii], npml, h, nu, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];

    
    
    # this step is a hack to avoid building new vectors in int32 every time
    for ii = 1:nSubdomains
        convert64_32!(SubArray[ii])
        
    end
    
    tic();
    for ii=1:nSubdomains
        factorize!(SubArray[ii])
        
    end
    println("Time for the factorization ", toc())
    
    Precond = PolarizedTracesPreconditioner(As, SubArray, nIt = 1 ,tol = 1e-6);    
    # we build a set of different incident waves
    
    theta = collect(1:0.3:2*pi)
    time = 0
    nit = 0
    for ii = 1:length(theta)
    
        u_inc = exp(k*im*(X*cos(theta[ii]) + Y*sin(theta[ii])));
        rhs = -(fastconv*u_inc - u_inc);
        
        tic();
        u = Precond\rhs
        time += toc();
    end
    
    
    println("Solving the sparse system with the positive perturbation via the method of polarized traces ")
    println("Frequency is ", k/(2*pi))
    println("Number of discretization points is ", 1/h)
    println("Number of Subdomains is ", nSubdomains)
    println("average time ", time/length(theta))
    println("npml points  ", npml )
    #println("average number of iterations ", nit/length(theta))
    #println("maximum number of inner iterations ", maxInnerIter )



end
