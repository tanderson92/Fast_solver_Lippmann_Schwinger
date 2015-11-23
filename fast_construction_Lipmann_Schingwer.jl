# small scrip to compute the solution of Lipman Schinwer equation
# We test the types introduced in FastConvolution.jl
# we test that the application is fast and that the construction
# is performed fast.


using PyPlot
using Devectorize


include("FastConvolution.jl")
include("quadratures.jl")
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

G = zeros(Complex128, N, N);

#nu(x,y) = 1 + x*0 + 0*y;

# we extract from a "tabulated" dictionary
# the good modification for the quadrature modification
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

tic();
for ii = 1:N
        r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
        r[ii] = 1;
        G[ii,:] =  1im/4*hankelh1(0, k*r)*h^2;
        G[ii,ii]= 1im/4*D0*h^2;
end
toc()

nu(x,y) = -0.3*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.4).*(abs(y).<0.4);
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



u_inc = exp(k*im*X);

#M = eye(N) + k^2*(G*spdiagm(nu(X,Y)));

#u = M\(-k^2*G*(spdiagm(nu(X,Y))*u_inc));

## Building the sparsifying preconditioner (this needs to be encapsulated)

# indVol = round(Integer, n*(n-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
# indVolC = setdiff(collect(1:N),indVol);
# Gcompressed = G[indVol,indVolC];
# GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC ];

# sum(abs(Gcompressed - GSampled))

# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# println(s)
# A = zeros(G);
# AcorrVol = (U[:,end])';
# ## Sparsifying preconditioner'

# indFz1 = round(Integer, n*(n-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
# indC = setdiff(collect(1:N),indFz1);
# GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
# Gcompressed = G[indFz1,indC]
# sum(abs(Gcompressed - GSampled))
# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# AcorrFz1 = (U[:,end])';

# #'
# indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
# indC = setdiff(collect(1:N),indFz2);
# GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];

# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# AcorrFz2 = (U[:,end])';


# ### ' for boundary in y

# indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
# indC = setdiff(collect(1:N),indFx1);
# GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];


# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# AcorrFx1 = (U[:,end])';


# #'
# indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
# indC = setdiff(collect(1:N),indFx2);
# GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];

# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# AcorrFx2 = (U[:,end])';

# # ' no for the corners
# indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
# indcorner2 = round(Integer, n + [0,-1, n,n-1]);
# indcorner3 = round(Integer, n^2-n+1 + [0,1, -n,-n+1]);
# indcorner4 = round(Integer, n^2 + [0,-1, -n,-n-1]);

# indC = setdiff(collect(1:N),indcorner1);
# GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# Acorrcorner1 = (U[:,end])';

# #'
# indC = setdiff(collect(1:N),indcorner2);
# GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# Acorrcorner2= (U[:,end])';

# #'
# indC = setdiff(collect(1:N),indcorner3);
# GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indC ];
# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# Acorrcorner3 = (U[:,end])';

# #'
# indC = setdiff(collect(1:N),indcorner4);
# GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indC ];
# # computing the sparsifying correction
# (U,s,V) = svd(GSampled);
# Acorrcorner4 = (U[:,end])';



# #' build the matrix (for loop)
# volInd = Int64[]

# for ii = 1:N
#     # take care of the corners first
#     if ii == 1
#         A[ii,indcorner1] = Acorrcorner1;
#     elseif ii == n
#         A[ii,indcorner2] = Acorrcorner2;
#     elseif ii == N-n+1
#         A[ii,indcorner3] = Acorrcorner3;
#     elseif ii == N
#         A[ii,indcorner4] = Acorrcorner4;
#     elseif ii > 1 && ii < n
#         ind  = ii + [-1,0,1,n,n+1, n-1];
#         A[ii,ind] = AcorrFx1;
#     elseif (ii > (N-n+1)) && ii < N
#         ind  = ii +[-1,0,1,-n,-n+1, -n-1];
#         A[ii,ind] = AcorrFx2;
#     elseif mod(ii,n) == 1
#         ind  = ii + [0,1,n,n+1,-n, -n+1];
#         A[ii,ind] = AcorrFz1;
#     elseif mod(ii,n) == 0
#         ind  = ii + [-1,0,n,n-1,-n, -n-1];
#         A[ii,ind] = AcorrFz2;
#     else
#         ind  = ii + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1];
#         A[ii,ind] = AcorrVol;
#         push!(volInd,ii)
#     end
# end


# bdyInd = setdiff(collect(1:N), volInd);

As = buildSparseA(k,X,Y,D0, n ,m);

# need to check everything here :S
Mapprox = zeros(Complex128, N,N);
Mapprox = As*G;
Mapproxsp = sparse(Mapprox.*(abs(As).>0));

Mapproxsp = k^2*(Mapproxsp*spdiagm(nu(X,Y)));
Mapproxsp = As + Mapproxsp;

Minv = lufact(Mapproxsp);

using IterativeSolvers

#M = 1
#G = 1
#gc()
# building the preconditioner

precond(x) = (Minv\(As*(fastconv*x)));

# building the RHS from the incident field
u_inc = exp(k*im*X);
rhs = -(fastconv*u_inc - u_inc);
#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


y= zeros(Complex128,N);
info =  gmres!(y, precond, Minv\(As*rhs))
info[2].residuals[:]
