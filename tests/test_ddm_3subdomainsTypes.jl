# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,


# Clean version of the DDM ode

using PyPlot
using Devectorize


include("../src/FastConvolution.jl")
include("../src/quadratures.jl")
include("../src/subdomains.jl")
#Defining Omega
h = 0.01
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

#G = buildConvMatrix(k,X,Y,D0,h)


#the domain is decomposed in two subdomains

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.46).*(abs(y).<0.46);

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


S1 = Subdomain(As,AG,Mapproxsp,x,y, 1, round(Integer, m/3)-1, 20, h, nu, k);
S2 = Subdomain(As,AG,Mapproxsp,x,y,  round(Integer, m/3), round(Integer, 2*m/3), 20, h, nu, k);
S3 = Subdomain(As,AG,Mapproxsp,x,y,  round(Integer, 2*m/3)+1, m, 20, h, nu, k);

# compute the right hand side
rhs = -(fastconv*u_inc - u_inc);
rhsA = (As*rhs);

# defining the Preconditioner
function precondIT(source)
    rhsA1 = source[S1.indVolInt]
    rhsA2 = source[S2.indVolInt]
    rhsA3 = source[S3.indVolInt]
    # put the rhs in the correct dumme vector
    @assert norm(source - vcat(rhsA1,rhsA2,rhsA3))<eps(1.0)

    rhsLocal1 = zeros(Complex128,S1.n*S1.m)
    rhsLocal2 = zeros(Complex128,S2.n*S2.m)
    rhsLocal3 = zeros(Complex128,S3.n*S3.m)

    rhsLocal1[S1.indVolIntLocal] = copy(rhsA1)
    rhsLocal2[S2.indVolIntLocal] = copy(rhsA2)
    rhsLocal3[S3.indVolIntLocal] = copy(rhsA3)

    # solve for each one

    u1 = solve(S1,rhsLocal1);
    u2 = solve(S2,rhsLocal2);
    u3 = solve(S3,rhsLocal3);

    #sweep up and down (just communicate trhoug the boundary)

    u1_N = u1[S1.ind_n];
    u1_N1 = u1[S1.ind_np];

    u2_0 = u2[S2.ind_0];
    u2_1 = u2[S2.ind_1];
    u2_N =  u2[S2.ind_n];
    u2_N1 = u2[S2.ind_np];

    u3_0 = u3[S3.ind_0];
    u3_1 = u3[S3.ind_1];


    #rhsLocal1[S1.ind_n]  += -S1.H[S1.ind_n,S1.ind_np]*u2_1
    #rhsLocal1[S1.ind_np] += S1.H[S1.ind_np,S1.ind_n]*u2_0

    # first sweep
    rhsLocal2[S2.ind_0] += S2.H[S2.ind_0,S2.ind_1]*u1_N1
    rhsLocal2[S2.ind_1] += -S2.H[S2.ind_1,S2.ind_0]*u1_N

    u2 = solve(S2,rhsLocal2);

    u22_N  =  u2[S2.ind_n];
    u22_N1 = u2[S2.ind_np];

    rhsLocal3[S3.ind_0] +=  S3.H[S3.ind_0,S3.ind_1]*u22_N1
    rhsLocal3[S3.ind_1] += -S3.H[S3.ind_1,S3.ind_0]*u22_N

    u3 = solve(S3,rhsLocal3);

    ##u3_0 = u3[S3.ind_0];
    ##u3_1 = u3[S3.ind_1];

    # backwards sweep
    rhsLocal2 = zeros(Complex128,S2.n*S2.m)
    rhsLocal2[S2.indVolIntLocal] = rhsA2

    rhsLocal2[S2.ind_n]  += -S2.H[S2.ind_n ,S2.ind_np]*u3_1
    rhsLocal2[S2.ind_np] +=  S2.H[S2.ind_np,S2.ind_n ]*u3_0

    u2 = solve(S2,rhsLocal2);

    u2_0 = u2[S2.ind_0];
    u2_1 = u2[S2.ind_1];

    rhsLocal2[S2.ind_0] += S2.H[S2.ind_0,S2.ind_1]*u1_N1
    rhsLocal2[S2.ind_1] += -S2.H[S2.ind_1,S2.ind_0]*u1_N

    u2 = solve(S2,rhsLocal2);

    rhsLocal1[S1.ind_n]  += -S1.H[S1.ind_n,S1.ind_np]*u2_1
    rhsLocal1[S1.ind_np] +=  S1.H[S1.ind_np,S1.ind_n]*u2_0

    u1 = solve(S1,rhsLocal1);


    return  vcat(u1[S1.indVolIntLocal],u2[S2.indVolIntLocal],u3[S3.indVolIntLocal]);

end

ApplyPrecond(x) = precondIT(Mapproxsp*x)

using IterativeSolvers

u = zeros(Complex128,N);
@time info =  gmres!(u, ApplyPrecond, precondIT(As*rhs), restart = 10)
println("number of iterations for inner solver is ", countnz(info[2].residuals[:]))

# solving the new


Minv = lufact(Mapproxsp);

function MinvAp(x)
    y = zeros(Complex128,N);
    info =  gmres!(y, ApplyPrecond, precondIT(x), tol = 1e-8, restart = 15)
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


