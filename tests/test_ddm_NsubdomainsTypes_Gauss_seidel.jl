# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,


# Clean version of the DDM ode

using PyPlot
using Devectorize
using IterativeSolvers

include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
#Defining Omega
h = 0.001
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

nSubdomains = 10;
println("Number of Subdomains is ", nSubdomains)
# we solve \triangle u + k^2(1 + nu(x))u = 0
# in particular we compute the scattering problem


# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

#G = buildConvMatrix(k,X,Y,D0,h)


#the domain is decomposed in two subdomains

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.48).*(abs(y).<0.48);

Ge = buildGConv(x,y,h,n,m,D0,k);
GFFT = fft(Ge);

fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

u_inc = exp(k*im*Y);

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

tic();
#SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], 20, h, nu, k, solvertype = "MKLPARDISO") for ii = 1:nSubdomains];
SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], 20, h, nu, k) for ii = 1:nSubdomains];
println("Time for the factorization ", toc())

# compute the right hand side
rhs = -(fastconv*u_inc - u_inc);
rhsA = (As*rhs);


# defining the Preconditioner (Gauss Seidel) (find the best way to anotate the array)
function precondGS(subDomains, source::Array{Complex128,1})
    ## We are only implementing the Jacobi version of the preconditioner
    nSubs = length(subDomains);

    n = subDomains[1].n
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].n*subDomains[ii].m) for ii = 1:nSubs ]

    # copying the wave-fields
    for ii = 1:nSubs
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]
    end

    uLocalArray = [solve(subDomains[ii],rhsLocal[ii] )  for  ii = 1:nSubs]

    u_0  = zeros(Complex128,n*nSubs)
    u_1  = zeros(Complex128,n*nSubs)
    u_n  = zeros(Complex128,n*nSubs)
    u_np = zeros(Complex128,n*nSubs)

    index = 1:n

    # Downward sweep

    u_n[index]  = uLocalArray[1][subDomains[1].ind_n]
    u_np[index] = uLocalArray[1][subDomains[1].ind_np]

    # populate the traces

    for ii = 1:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        if ii != 1
            u_0[(ii-1)*n + index] = -uLocalArray[ii][ind_0]
            u_1[(ii-1)*n + index] = -uLocalArray[ii][ind_1]
        end
        if ii !=nSubs
            u_n[(ii-1)*n  + index] = -uLocalArray[ii][ind_n]
            u_np[(ii-1)*n + index] = -uLocalArray[ii][ind_np]
        end

    end


    u_n[ index] = -u_n[ index]
    u_np[index] = -u_np[index]

    for ii = 2:nSubs-1
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_0] =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1] = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]

        uDown = solve(subDomains[ii],rhsLocaltemp)

        u_n[(ii-1)*n  + index] = -u_n[(ii-1)*n  + index] + uDown[ind_n]
        u_np[(ii-1)*n + index] = -u_np[(ii-1)*n + index] + uDown[ind_np]

    end

    #Applying L


    for ii = 2:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        if ii != nSubs

        rhsLocaltemp[subDomains[ii].ind_0]  =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1]  = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]


        rhsLocaltemp[subDomains[ii].ind_n]  = -subDomains[ii].H[ind_n,ind_np]*u_np[(ii-1)*n + index]
        rhsLocaltemp[subDomains[ii].ind_np] =  subDomains[ii].H[ind_np,ind_n]*u_n[(ii-1)*n + index]

        else
            rhsLocaltemp[subDomains[ii].ind_0]  =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_1]  = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
        end

        uL = solve(subDomains[ii],rhsLocaltemp)

        # saving the reflections
        u_0[(ii-1)*n + index] -= uL[ind_0]
        u_1[(ii-1)*n + index] -= uL[ind_1] - u_np[(ii-2)*n + index]

    end

    # Upward sweep

     u_0[(nSubs-1)*n + index] = -u_0[(nSubs-1)*n + index]
     u_1[(nSubs-1)*n + index] = -u_1[(nSubs-1)*n + index]

     for ii = nSubs-1:-1:2
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source (we may need a sparse source)
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_np] =  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
        rhsLocaltemp[subDomains[ii].ind_n]  = -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]

        uUp = solve(subDomains[ii],rhsLocaltemp)

        u_0[(ii-1)*n + index] = -u_0[(ii-1)*n + index] + uUp[ind_0]
        u_1[(ii-1)*n + index] = -u_1[(ii-1)*n + index] + uUp[ind_1]

    end

    # reconstruction

    uPrecond = Complex128[]

    for ii = 1:nSubs

        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        # adding the source at the boundaries
        if ii!= 1
            # we need to be carefull at the edges
            rhsLocaltemp[subDomains[ii].ind_1]  += -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_0]  +=  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]

        end
        if ii!= nSubs
            rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
            rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]
        end

        uLocal = solve(subDomains[ii],rhsLocaltemp)

        uPrecond = vcat(uPrecond,uLocal[subDomains[ii].indVolIntLocal])
    end
    return  uPrecond
end


ApplyPrecondGS(x) = precondGS(SubArray, Mapproxsp*x)

u = zeros(Complex128,N);
@time info =  gmres!(u, ApplyPrecondGS, precondGS(SubArray, As*rhs), restart = 20)
println("number of iterations for inner solver using a Gauss-Seidel preconditioner is ", countnz(info[2].residuals[:]))

# solving the new

Minv = lufact(Mapproxsp);

function MinvAp(x)
    y = zeros(Complex128,N);
    info =  gmres!(y, ApplyPrecondGS, precondGS(SubArray,x), tol = 1e-4, restart = 40)
    println("number of iterations for inner solver a Gauss-Seidel preconditioner is ", countnz(info[2].residuals[:]))
    return y
end

# random vector
rhs = rand(N) + im*rand(N);

norm(Minv \ rhs - MinvAp(rhs))/norm(rhs)

precond(x) = Minv\(As*(fastconv*x));

# building the RHS from the incident field
u_inc = exp(k*im*Y);
rhs = -(fastconv*u_inc - u_inc);
#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


# u = zeros(Complex128,N);
# @time info =  gmres!(u, precond, Minv\(As*rhs))
# println(info[2].residuals[:])


