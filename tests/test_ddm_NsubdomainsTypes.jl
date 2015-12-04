# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,


# Clean version of the DDM ode

using PyPlot
using Devectorize


include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
#Defining Omega
h = 0.001
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

nSubdomains = 20;
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

As = buildSparseA(k,X,Y,D0, n ,m);

AG = buildSparseAG(k,X,Y,D0, n ,m);

# need to check everything here :S

Mapproxsp = k^2*(AG*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

# number of interior points for each subdomain
SubDomLimits = round(Integer, floor(linspace(1,m+1,nSubdomains+1)))
# index in y. of the first row of each subdomain
idx1 = SubDomLimits[1:end-1]
# index in y of the last row of each subdomains
idxn = SubDomLimits[2:end]-1


SubArray = [ Subdomain(As,AG,Mapproxsp,x,y, idx1[ii] , idxn[ii], 50, h, nu, k) for ii = 1:nSubdomains];

# compute the right hand side
rhs = -(fastconv*u_inc - u_inc);
rhsA = (As*rhs);

# defining the Preconditioner
function precondIT(subDomains, source)
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

    for ii = 2:nSubs-1
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        rhsLocaltemp[subDomains[ii].ind_0] +=  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1] += -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]

        uDown = solve(subDomains[ii],rhsLocaltemp)

        u_n[(ii-1)*n  + index] = uDown[ind_n]
        u_np[(ii-1)*n + index] = uDown[ind_np]

    end

    # Upward sweep

    u_0[(nSubs-1)*n + index] = uLocalArray[end][subDomains[end].ind_0]
    u_1[(nSubs-1)*n + index] = uLocalArray[end][subDomains[end].ind_1]


     for ii = nSubs-1:-1:2
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]
        rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]

        uUp = solve(subDomains[ii],rhsLocaltemp)

        u_0[(ii-1)*n + index] = uUp[ind_0]
        u_1[(ii-1)*n + index] = uUp[ind_1]

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
            rhsLocaltemp[subDomains[ii].ind_0]  +=  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_1]  += -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
        end
        if ii!= nSubs
            rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]
            rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
        end

        uLocal = solve(subDomains[ii],rhsLocaltemp)

        uPrecond = vcat(uPrecond,uLocal[subDomains[ii].indVolIntLocal])
    end
    return  uPrecond
end

ApplyPrecond(x) = precondIT(SubArray, Mapproxsp*x)

using IterativeSolvers

u = zeros(Complex128,N);
@time info =  gmres!(u, ApplyPrecond, precondIT(SubArray, As*rhs), restart = 20)
println("number of iterations for inner solver is ", countnz(info[2].residuals[:]))

# solving the new


Minv = lufact(Mapproxsp);

function MinvAp(x)
    y = zeros(Complex128,N);
    info =  gmres!(y, ApplyPrecond, precondIT(SubArray,x), tol = 1e-8, restart = 40)
    println("number of iterations for inner solver is ", countnz(info[2].residuals[:]))
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


u = zeros(Complex128,N);
@time info =  gmres!(u, precond, Minv\(As*rhs))
println(info[2].residuals[:])


