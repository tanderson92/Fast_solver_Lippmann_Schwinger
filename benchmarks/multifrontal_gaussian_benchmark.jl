# script to benchmark the multifrontal solver 


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
# H = [ 0.005, 0.002,0.0025, 0.00125, 0.001 0.000625]
# Subs = [ 4,8,10, 16,20,32]

# NPML = [ 10, 12, 14, 16]

H = [ 0.005, 0.0025, 0.002, 0.00125, 0.001, 0.0008, 0.0007142857142857143, 0.000625, 0.0005]



## Definin the wave speed 

nu(x,y)=   -(1.200*exp(-8000*((x-0.33).^2 + (y+0.41).^2))  +
            1.100*exp(-8000*((x+0.15).^2 + (y+0.39).^2))  +
            1.100*exp(-8000*((x-0.06).^2 + (y-0.05).^2))  +
            1.140*exp(-8000*((x+0.43).^2 + (y+0.10).^2))  +
            1.200*exp(-8000*((x-0.23).^2 + (y+0.12).^2))  +
            1.200*exp(-8000*((x-0.24).^2 + (y+0.29).^2))  +
            1.230*exp(-8000*((x-0.41).^2 + (y+0.00).^2))  +
            1.250*exp(-8000*((x+0.11).^2 + (y+0.11).^2))  +
            1.137*exp(-8000*((x+0.29).^2 + (y-0.05).^2))  +
            1.050*exp(-8000*((x-0.01).^2 + (y+0.31).^2))  +
            1.121*exp(-8000*((x-0.04).^2 + (y-0.41).^2))  +
            1.140*exp(-8000*((x-0.30).^2 + (y+0.21).^2))  +
            1.148*exp(-8000*((x+0.32).^2 + (y+0.43).^2))  +
            1.037*exp(-8000*((x-0.08).^2 + (y+0.02).^2))  +
            1.150*exp(-8000*((x+0.40).^2 + (y-0.02).^2))  +
            1.121*exp(-8000*((x+0.36).^2 + (y+0.14).^2))  +
            1.100*exp(-8000*((x+0.36).^2 + (y-0.14).^2))  +
            1.100*exp(-8000*((x-0.04).^2 + (y-0.07).^2))  +
            1.100*exp(-8000*((x+0.07).^2 + (y-0.43).^2))  +
            1.100*exp(-8000*((x-0.42).^2 + (y-0.22).^2))  +
            1.200*exp(-8000*((x-0.37).^2 + (y-0.09).^2))  +
            1.230*exp(-8000*((x-0.22).^2 + (y+0.26).^2))  +
            1.250*exp(-8000*((x-0.08).^2 + (y+0.40).^2))  +
            1.137*exp(-8000*((x-0.38).^2 + (y-0.13).^2))  +
            1.150*exp(-8000*((x+0.27).^2 + (y+0.28).^2))  +
            1.121*exp(-8000*((x+0.26).^2 + (y-0.28).^2))  +
            1.140*exp(-8000*((x+0.18).^2 + (y-0.19).^2))  +
            1.148*exp(-8000*((x+0.27).^2 + (y-0.16).^2))  +
            1.037*exp(-8000*((x+0.18).^2 + (y+0.08).^2))  +
            1.150*exp(-8000*((x-0.40).^2 + (y+0.09).^2))  +
            1.121*exp(-8000*((x-0.22).^2 + (y+0.04).^2))  +
            1.140*exp(-8000*((x-0.16).^2 + (y+0.32).^2))  +
            1.100*exp(-8000*((x+0.40).^2 + (y+0.23).^2))  +
            1.100*exp(-8000*((x+0.23).^2 + (y+0.10).^2))  +
            1.140*exp(-8000*((x+0.13).^2 + (y-0.23).^2))  +
            1.200*exp(-8000*((x-0.00).^2 + (y+0.02).^2))  +
            1.200*exp(-8000*((x-0.32).^2 + (y-0.21).^2))  +
            1.230*exp(-8000*((x-0.12).^2 + (y+0.13).^2))  +
            1.250*exp(-8000*((x+0.33).^2 + (y-0.00).^2))  +
            1.137*exp(-8000*((x+0.00).^2 + (y-0.08).^2))  +
            1.150*exp(-8000*((x-0.32).^2 + (y-0.13).^2))  +
            1.121*exp(-8000*((x-0.14).^2 + (y-0.02).^2))  +
            1.140*exp(-8000*((x-0.43).^2 + (y+0.19).^2))  +
            1.148*exp(-8000*((x+0.18).^2 + (y+0.06).^2))  +
            1.037*exp(-8000*((x-0.37).^2 + (y+0.09).^2))  +
            1.150*exp(-8000*((x+0.30).^2 + (y-0.25).^2))  +
            1.221*exp(-8000*((x+0.20).^2 + (y+0.39).^2))  +
            1.100*exp(-8000*((x+0.13).^2 + (y+0.33).^2))  +
            1.100*exp(-8000*((x-0.39).^2 + (y-0.31).^2))  +
            1.100*exp(-8000*((x+0.11).^2 + (y-0.13).^2))  +
            1.100*exp(-8000*((x-0.20).^2 + (y-0.44).^2))  +
            1.200*exp(-8000*((x-0.08).^2 + (y-0.09).^2))  +
            1.230*exp(-8000*((x-0.24).^2 + (y+0.24).^2))  +
            1.250*exp(-8000*((x-0.00).^2 + (y+0.41).^2))  +
            1.137*exp(-8000*((x-0.27).^2 + (y-0.18).^2))  +
            1.150*exp(-8000*((x+0.41).^2 + (y+0.28).^2))  +
            1.221*exp(-8000*((x+0.16).^2 + (y-0.33).^2))  +
            1.140*exp(-8000*((x+0.22).^2 + (y-0.04).^2))  +
            1.248*exp(-8000*((x+0.41).^2 + (y-0.07).^2))  +
            1.137*exp(-8000*((x+0.17).^2 + (y+0.30).^2))  +
            1.150*exp(-8000*((x-0.14).^2 + (y+0.01).^2))  +
            1.021*exp(-8000*((x-0.39).^2 + (y+0.33).^2))  +
            1.140*exp(-8000*((x-0.12).^2 + (y+0.26).^2))  +
            1.048*exp(-8000*((x-0.09).^2 + (y-0.03).^2))).*(abs(x).<0.48).*(abs(y).<0.48);

# and its transpose
    nuT(x,y)= nu(y,x)



# for kk = 1:length(tolerance)
for ll = 1:length(H)

    # setting the grid space
    h = H[ll]

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
    
    
    # we solve \triangle u + k^2(1 + nu(x))u = 0
    # in particular we compute the scattering problem
    
    
    # we extract from a "tabulated" dictionary
    # the good modification for the quadrature modification
    (ppw,D) = referenceValsTrapRule();
    D0 = D[1];
    
    
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
    
    println("Building the A sparse")
    @time As = buildSparseA(k,X,Y,D0, n ,m);
    
    println("Building A*G in sparse format")
    @time AG = buildSparseAG(k,X,Y,D0, n ,m);

    # defining the preconditioner
    IntPrecond =  SparsifyingPreconditioner(Mapproxsp, As);
    
    # loop to test for different tolerances

            # we build a set of different incident waves
            
            theta = collect(1:0.3:2*pi)
            time = 0
            for ii = 1:length(theta)
            
                u_inc = exp(k*im*(X*cos(theta[ii]) + Y*sin(theta[ii])));
                rhs = -(fastconv*u_inc - u_inc);
            
                u = zeros(Complex128,N);
                tic();
                info = gmres!(u, fastconv, rhs, IntPrecond, tol = 1.e-10)
                time += toc();
            end
            
            println("Solving the gaussian bumps wavespeed using multifrontal solver")
            println("Frequency is ", k/(2*pi))
            println("Number of discretization points is ", 1/h)
            println("Number of Subdomains is ", 1)
            println("average time ", time/length(theta))
        end
    end
    
end
