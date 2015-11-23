# small script to test how to perform the domain decomposition
# we try to compute fast solution without tampering with A directly,
# we only modify nu and As
using PyPlot
using Devectorize


include("FastConvolution.jl")
include("quadratures.jl")
#Defining Omega
h = 0.005
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

G = buildConvMatrix(k,X,Y,D0,h)


x1 = -a/2:h:a/2
y1 = y[1:round(Integer, ceil(3*end/4))]
n1 = length(x1)
m1 = length(y1)
N1 = n1*m1
X1 = repmat(x1, 1, m1)[:]
Y1 = repmat(y1', n1,1)[:]

x2 = -a/2:h:a/2
y2 = y[round(Integer, ceil(end/4)):end]
n2 = length(x2)
m2 = length(y2)
N2 = n2*m2
X2 = repmat(x2, 1, m2)[:]
Y2 = repmat(y2', n2,1)[:]


#the domain is decomposed in two subdomains

filter1(x,y) = 0*x + (y.<=2*h) + (y.>2*h).*(y.<y1[end-2]).*exp(-100*(y-2*h).^2)
filter2(x,y) = 0*x + (y.>=-h) + (y.<-h).*(y.>y2[3]).*exp(-100*(y+h).^2)

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.4).*(abs(y).<0.4);

nu1(x,y) = filter1(x,y).*nu(x,y);
nu2(x,y) = filter2(x,y).*nu(x,y);


#nu(x,y) = ((abs(x)+abs(y)).<0.2)

xe = collect((x[1]-(n-1)*h):h:(x[end]+(n-1)*h));
ye = collect((y[1]-(m-1)*h):h:(y[end]+(m-1)*h));

Xe = repmat(xe, 1, 3*m-2);
Ye = repmat(ye', 3*n-2,1);

R = sqrt(Xe.^2 + Ye.^2);
# to avoid evaluating at the singularity
indMiddle = round(Integer, m-1 + (m+1)/2)
R[indMiddle,indMiddle] =1;
Ge = 1im/4*hankelh1(0, k*R)*h^2;
Ge[indMiddle,indMiddle] = 1im/4* D0*h^2;

GFFT = fft(Ge);
fastconv = FastM(GFFT,nu(X,Y),3*n-2,3*m-2,n, m, k);

u_inc = exp(k*im*Y);

# bdyInd = setdiff(collect(1:N), volInd);

As = buildSparseA(k,X,Y,D0, n ,m);

# need to check everything here :S
Mapprox = zeros(Complex128, N,N);
Mapprox = As*G;
Mapproxtemp = sparse(Mapprox.*(abs(As).>0));

Mapproxsp = k^2*(Mapproxtemp*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

indBdy1 = find(Y.==0);
indBdy2 = find(Y.==h);

indVolInt1 =  find(Y.< h)
indVol1   =  find(Y.<= y1[end])
indVolIntLocal1 =  find(Y1.< h)

Mapproxsp1 =  (As + k^2*(Mapproxtemp*spdiagm(nu1(X,Y))))[indVol1,indVol1];
Mapproxsp1[end-2*n+1:end, end-2*n+1:end]=Mapproxsp[end-2*n+1:end, end-2*n+1:end];


indVolInt2 =  find(Y.> 0)
indVol2   =  find(Y.>= y2[1])
indVolIntLocal2 =  find(Y2.> 0)
Mapproxsp2 =  (As + k^2*(Mapproxtemp*spdiagm(nu2(X,Y))))[indVol2,indVol2];
Mapproxsp2[1:2*n, 1:2*n]=Mapproxsp[1:2*n, 1:2*n];


ind1_N = find(Y1.==0);
ind1_N1 = find(Y1.==h);

ind2_0 = find(Y2.==0);
ind2_1 = find(Y2.==h);

# invert the local matrices

M1inv = lufact(Mapproxsp1);
M2inv = lufact(Mapproxsp2);

## testing local GReen's function by changing the boundary conditions

dirac = zeros(Complex128, N)
dirac[n*(n+1)/2 + (n+1)/2] = 1;

diracA = As*dirac;

dirac1 = zeros(Complex128,N1)

dirac1[indVolIntLocal1] = diracA[indVolIntLocal1]




# compute the right hand side
rhs = -(fastconv*u_inc - u_inc);
rhsA = (As*rhs);
# we need to split the rhs in two parts
rhsA1 = rhsA[indVolInt1]
rhsA2 = rhsA[indVolInt2]
# put the rhs in the correct dumme vector

rhsLocal1 = zeros(Complex128,N1)
rhsLocal2 = zeros(Complex128,N2)

rhsLocal1[indVolIntLocal1] = rhsA1
rhsLocal2[indVolIntLocal2] = rhsA2

# solve for each one

u1 = M1inv\rhsLocal1;
u2 = M2inv\rhsLocal2;

#sweep up and down (just communicate trhoug the boundary)

u1_N = u1[ind1_N];
u1_N1 = u1[ind1_N1];

u2_0 = u2[ind2_0];
u2_1 = u2[ind2_1];

rhsLocal1[ind1_N]  += -Mapproxsp1[ind1_N,ind1_N1]*u2_1
rhsLocal1[ind1_N1] += Mapproxsp1[ind1_N1,ind1_N]*u2_0

rhsLocal2[ind2_0] += Mapproxsp2[ind2_0,ind2_1]*u1_N1
rhsLocal2[ind2_1] += -Mapproxsp2[ind2_1,ind2_0]*u1_N

u11 = M1inv\rhsLocal1;
u21 = M2inv\rhsLocal2;

uPrecond = vcat(u11[indVolIntLocal1],u21[indVolIntLocal2]);

# testing the recovery of the good solution


Minv = lufact(Mapproxsp);

uref = Minv\(As*rhs);

rhsLocal1 = zeros(Complex128,N1)
rhsLocal2 = zeros(Complex128,N2)

rhsLocal1[indVolIntLocal1] = rhsA1
rhsLocal2[indVolIntLocal2] = rhsA2

indBdy0 = find(Y.==0)
indBdy1 = find(Y.==h)


rhsLocal1[ind1_N]  += -Mapproxsp1[ind1_N,ind1_N1]*uref[indBdy1]
rhsLocal1[ind1_N1] += Mapproxsp1[ind1_N1,ind1_N]*uref[indBdy0]

rhsLocal2[ind2_0] += Mapproxsp2[ind2_0,ind2_1]*uref[indBdy1]
rhsLocal2[ind2_1] += -Mapproxsp2[ind2_1,ind2_0]*uref[indBdy0]

u11 = M1inv\rhsLocal1;
u21 = M2inv\rhsLocal2;


uReconstruct = vcat(u11[indVolIntLocal1],u21[indVolIntLocal2]);

residual = Mapproxsp*uReconstruct - rhsA

println("residual of the reconstructed solution is :", sum(abs(residual)))

# testing the recoveru with the global function

rhstest = zeros(rhsA);
rhstest[indVolIntLocal1] = rhsA1;
rhstest[ind1_N]  += -Mapproxsp[ind1_N,ind1_N1]*uref[indBdy1]
rhstest[ind1_N1] += Mapproxsp[ind1_N1,ind1_N]*uref[indBdy0]


uref1 = Minv\(rhstest);


function precondIT(source)
rhsA1 = source[indVolInt1]
rhsA2 = source[indVolInt2]
# put the rhs in the correct dumme vector

rhsLocal1 = zeros(Complex128,N1)
rhsLocal2 = zeros(Complex128,N2)

rhsLocal1[indVolIntLocal1] = rhsA1
rhsLocal2[indVolIntLocal2] = rhsA2

# solve for each one

u1 = M1inv\rhsLocal1;
u2 = M2inv\rhsLocal2;

#sweep up and down (just communicate trhoug the boundary)

u1_N = u1[ind1_N];
u1_N1 = u1[ind1_N1];

u2_0 = u2[ind2_0];
u2_1 = u2[ind2_1];


rhsLocal1[ind1_N]  += -Mapproxsp1[ind1_N,ind1_N1]*u2_1
rhsLocal1[ind1_N1] += Mapproxsp1[ind1_N1,ind1_N]*u2_0

rhsLocal2[ind2_0] += Mapproxsp2[ind2_0,ind2_1]*u1_N1
rhsLocal2[ind2_1] += -Mapproxsp2[ind2_1,ind2_0]*u1_N


u11 = M1inv\rhsLocal1;
u21 = M2inv\rhsLocal2;

    return  vcat(u11[indVolIntLocal1],u21[indVolIntLocal2]);

end

ApplyPrecond(x) = Mapproxsp*(precondIT(x))

using IterativeSolvers


u= zeros(Complex128,N);
@time info =  gmres!(u, ApplyPrecond, precondIT(As*rhs))
prinln(info[2].residuals[:])



residual = Mapproxsp*uPrecond;
# solving the new

Minv = lufact(Mapproxsp);


#M = 1
#G = 1
#gc()
# building the preconditioner

precond(x) = (Minv\(As*(fastconv*x)));

# building the RHS from the incident field
u_inc = exp(k*im*Y);
rhs = -(fastconv*u_inc - u_inc);
#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


u= zeros(Complex128,N);
@time info =  gmres!(u, precond, Minv\(As*rhs))
info[2].residuals[:]
