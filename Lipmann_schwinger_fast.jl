# small scrip to compute the solution of Lipman Schinwer equation
# We optimize the construction of the Lipman Schingwer equation
# G is never formed explicitely we apply a Fourier Transform


using PyPlot
using Devectorize



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

G = zeros(Complex128, N, N)

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

nu(x,y) = - 0.5*exp(-20*(x.^2 + y.^2)).*(abs(x).<0.4).*(abs(y).<0.4);
#nu(x,y) = ((abs(x)+abs(y)).<0.2)

xe = collect((x[1]-(n-1)*h):h:(x[end]+(n-1)*h))
ye = collect((y[1]-(n-1)*h):h:(y[end]+(n-1)*h))


Xe = repmat(xe, 1, 3*m-2)
Ye = repmat(ye', 3*n-2,1)

R = sqrt(Xe.^2 + Ye.^2);
# to avoid evaluating at the singularity
R[round(Integer,ceil(end/2)),round(Integer,ceil(end/2))] =1;
Ge = 1im/4*hankelh1(0, k*R)*h^2;
Ge[round(Integer,ceil(end/2)),round(Integer,ceil(end/2))] = 1im/4*D0*h^2;

btest = rand(N) + im*rand(N);

Btest = reshape(btest, n,n);

BFT = zeros(3*m-2, 3*m-2);

BFT[1:m,1:m]= Btest ;



GFFT = fft(Ge);
BFFT  = fft(BFT);
YFFT = GFFT.*BFFT;
YFT = ifft(YFFT);

YT = YFT[round(Integer,((end+1)/2)): round(Integer,((end+1)/2)+m-1), round(Integer,((end+1)/2)): round(Integer,((end+1)/2)+m-1)];

ytest = YT[:];

yref = G*btest;

println(norm(ytest-yref))

u_inc = exp(k*im*X);

M = eye(N) + k^2*(G*spdiagm(nu(X,Y)));

rhs = (-k^2*G*(spdiagm(nu(X,Y))*u_inc));
#u = M\(-k^2*G*(spdiagm(nu(X,Y))*u_inc));

## Building the sparsifying preconditioner

indVol = round(Integer, n*(n-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
indVolC = setdiff(collect(1:N),indVol);
Gcompressed = G[indVol,indVolC];

# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
println(s)
A = zeros(G);
AcorrVol = (U[:,end])';
## Sparsifying preconditioner '


indFz1 = round(Integer, n*(n-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
indC = setdiff(collect(1:N),indFz1);
Gcompressed = G[indFz1,indC]

# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
AcorrFz1 = (U[:,end])';


indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
indC = setdiff(collect(1:N),indFz2);
Gcompressed = G[indFz2,indC]

# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
AcorrFz2 = (U[:,end])';


### for boundary in y

indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
indC = setdiff(collect(1:N),indFx1);
Gcompressed = G[indFx1,indC]

# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
AcorrFx1 = (U[:,end])';

#'
indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
indC = setdiff(collect(1:N),indFx2);
Gcompressed = G[indFx2,indC]

# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
AcorrFx2 = (U[:,end])';

indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
indcorner2 = round(Integer, n + [0,-1, n,n-1]);
indcorner3 = round(Integer, n^2-n+1 + [0,1, -n,-n+1]);
indcorner4 = round(Integer, n^2 + [0,-1, -n,-n-1]);

indC = setdiff(collect(1:N),indcorner1);
Gcompressed = G[indcorner1,indC];
# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
Acorrcorner1 = (U[:,end])';

indC = setdiff(collect(1:N),indcorner2);
Gcompressed = G[indcorner2,indC];
# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
Acorrcorner2= (U[:,end])';

indC = setdiff(collect(1:N),indcorner3);
Gcompressed = G[indcorner3,indC];
# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
Acorrcorner3 = (U[:,end])';

indC = setdiff(collect(1:N),indcorner4);
Gcompressed = G[indcorner4,indC];
# computing the sparsifying correction
(U,s,V) = svd(Gcompressed);
Acorrcorner4 = (U[:,end])';



#' build the matrix (for loop)
volInd = Int64[]

for ii = 1:N
    # take care of the corners first
    if ii == 1
        A[ii,indcorner1] = Acorrcorner1;
    elseif ii == n
        A[ii,indcorner2] = Acorrcorner2;
    elseif ii == N-n+1
        A[ii,indcorner3] = Acorrcorner3;
    elseif ii == N
        A[ii,indcorner4] = Acorrcorner4;
    elseif ii > 1 && ii < n
        ind  = ii + [-1,0,1,n,n+1, n-1];
        A[ii,ind] = AcorrFx1;
    elseif (ii > (N-n+1)) && ii < N
        ind  = ii +[-1,0,1,-n,-n+1, -n-1];
        A[ii,ind] = AcorrFx2;
    elseif mod(ii,n) == 1
        ind  = ii + [0,1,n,n+1,-n, -n+1];
        A[ii,ind] = AcorrFz1;
    elseif mod(ii,n) == 0
        ind  = ii + [-1,0,n,n-1,-n, -n-1];
        A[ii,ind] = AcorrFz2;
    else
        ind  = ii + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1];
        A[ii,ind] = AcorrVol;
        push!(volInd,ii)
    end
end


bdyInd = setdiff(collect(1:N), volInd);

As = sparse(A);
#C = A*G;
#CC = C.*(abs(A).>0);

#CC = sparse(CC);

# B = A[bdyInd.:]*G*(spdiagm(nu(X,Y))

# need to check everything here :S
Mapprox = zeros(Complex128, N,N);
Mapprox = As*M;

#Mapproxsp = sparse(Mapprox.*(abs(A).>0.0002));

Mapproxsp = sparse(Mapprox.*(abs(A).>0));

Minv = lufact(Mapproxsp);

using IterativeSolvers

precond(x) = (Minv\(As*(M*x)));

#x = zeros(Complex128,N);
#info =  gmres!(x, M, rhs, maxiter  = 60)


y= zeros(Complex128,N);
info =  gmres!(y, precond, Minv\(As*rhs))
info[2].residuals[:]
