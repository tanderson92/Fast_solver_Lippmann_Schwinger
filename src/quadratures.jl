
using GSL

function gaussLegendreQuad(N::Int64,a::Float64 ,b::Float64)
    # Function to compute the Gauss-Legendre quadratures
    # modified from Greg von Winckel - 02/25/2004

    n  = N-1;
    n1 = n+1;
    n2 = n+2;

    xu = linspace(-1, 1, n1)

    # Initial guess
    y = cos((2*(0:n)'+1)*pi/(2*n+2))+(0.27/n1)*sin(pi*xu.'*n/n2)
    y = y[:]

    # Derivative of LGVM
    Lp=zeros(n1,1);

    # initial condition
    y0 = 2;

    # Newton-Rhapson to compute he zeros
    while maximum(abs(y-y0))>eps(1.0)

        L = evalLegendrePol(n2,y)
        # computing the derivative
        Lp=(n2)*( L[:,n1]-y.*L[:,n2] )./(1-y.^2);

        y0 = y;
        y  = y0-L[:,n2]./Lp;

    end
    # Linear map from[-1,1] to [a,b]
    x = (a*(1-y)+b*(1+y))/2;

    # Compute the weights
    w = (b-a)./((1-y.^2).*Lp.^2)*(n2/n1)^2;

    return (x,w)

end

function evalLegendrePol(M::Int64,x::Array{Float64,1})
    VandM = zeros(length(x)[1], M)
    # setting the initial conditions
    VandM[:,1] = 1
    VandM[:,2] = x

    # Running the recursion
    for k=2:M-1
        VandM[:,k+1] = ((2*k-1)*x.*VandM[:,k]-(k-1)*VandM[:,k-1] )/k;
    end

    return VandM
end

function referenceValsTrapRule()
    x = 2.0.^(-(0:5))[:]
    w = [1-0.892*im, 1-1.35*im, 1-1.79*im, 1- 2.23*im, 1-2.67*im, 1-3.11*im]
    return (x,w)
end



function evalPhi(M::Int64,y::Float64,x::Array{Float64,1}; weak = true, singular = true, hyper = false )
    pxi = evalLegendrePol(M,x)'
    # the fisr part with only the polynomials
    Phi = pxi
    YminusX = repmat((y-x)', M,1)
    if weak
        # polynomials times a logarithm
        Phi = [Phi; pxi.*( log(abs(YminusX)) )]
    end
    if singular
    # polynomials times 1/(y-x)
    Phi = [Phi; pxi.*( 1./(YminusX)) ]
    end
    if hyper
    # polynomials times 1/(y-x).^2
    Phi = [Phi; pxi.*( 1./(YminusX).^2) ]
    end
    # if we ever run into a singularity (just remove it)
    Phi[Phi.==Inf] = 0

    return Phi
end

function generalizedGaussianQuad(M::Int64,y::Float64,x::Array{Float64,1}, w::Array{Float64,1} )
    # y can not be in the same panel
    @assert abs(y) > 1
    Phi = evalPhi(M,y,x)

    end

# We provide the modified weights to compute a high order quadrature to integrate
# smooth functions against different
function modifiedGLWeights1( w::Array{Float64,1},x::Array{Float64,1},y::Float64 )
    # function to obtain the modifed Gauss-Legendre weights to integrate
    # pv \int_{-1}^1 P_n(x)/(y-x)
    # Eq 71 in Numerical Quadratures for Singular and Hypersingular Integrals
    N = length(x)
    M = evalLegendrePol(N,x)
    # evaluating the Legendre function of second kind using GSL
    # Q = Q_j(y)
    Q = sf_legendre_Ql(repmat((0:N-1)',N,1 ), repmat(y*ones(N,1),1,N ));
    J = repmat(2*(0:N-1)'+1,N,1 )
    return  (w.*(sum(M.*Q.*J,2)))[:]
end

function modifiedGLWeights2( w::Array{Float64,1},x::Array{Float64,1},y::Float64 )
    # function to obtain the modifed Gauss-Legendre weights to integrate
    # \int_{-1}^1 P_n(x) 0.5*log((y-x)^2)
    # Eq 72 in Numerical Quadratures for Singular and Hypersingular Integrals
    N = length(x)
    # we evaluate the the Legendre polynomial up to order N-1
    # at the quadrature points
    M = evalLegendrePol(N,x)
    # We  compute the correction for the weights
    # evaluating the Legendre function of second kind using GSL
    # P_0(x_n) - P_1(x_n) *R_0(y)
    Wcorr1 = (M[:,1]-M[:,2]).*R_j(zeros(Int64,N,1),y*ones(N,1))
    # \sum_{j=1}^{N-2} (P_{j-1}(x_n) - P_{j+1}(x_n))*R_j(y)
    Wcorr2 = (M[:,1:N-2]-M[:,3:N]).*(R_j(repmat((1:N-2)',N,1), repmat(y*ones(N,1),1,N-2)) ) ;
    # P_{N-2}(x_n)*R_{N-1}(y)
    Wcorr3  = M[:,N-1].*R_j((N-1)*ones(Int64, N,1),y*ones(N,1))
    # P_{N-1}(x_n)*R_{N}(y)
    Wcorr4  = M[:,N].*R_j((N)*ones(Int64, N,1),y*ones(N,1))
    # adding everything up
    Wcorr = Wcorr1 + sum(Wcorr2,2) + Wcorr3 + Wcorr4;
    return  (w.*Wcorr)[:]
end

function computeweights(M::Int64, x::Array{Float64,1}, w::Array{Float64,1}, y::Float64)
    Phi = evalPhi(M,y,x)
    m1 = sum(Phi[1:M,:]*diagm(w),2)
    # modified weights for the logarithmic singularity
    w1 = modifiedGLWeights1( w,x,y)
    m2 = sum(Phi[M+1:2*M,:]*diagm(w1),2)
    w2 = modifiedGLWeights2( w,x,y)
    m3 = sum(Phi[2*M+1:3*M,:]*diagm(w2),2)
    m  = vcat(m1,m2,m3)
    wMod = Phi\m
end

function legendreInterpMatrix( s::Array{Float64,1}, x::Array{Float64,1}, w::Array{Float64,1})
    # interpolation matrix from Gauss-Legendre points x to the target points s
    n = length(x);
    LegendrePol = evalLegendrePol(n,x)
    LegendreInterp = evalLegendrePol(n,s)
    return  LegendreInterp*(diagm((2*(0:n-1)+1)/2))*(LegendrePol.')*diagm(w)
end

function modifiedGLWeights3( w::Array{Float64,1},x::Array{Float64,1},y::Float64 )
    # function to obtain the modifed Gauss-Legendre weights to integrate
    # \int_{-1}^1 P_n(x) /(y-x).^2
    # Eq 73 in Numerical Quadratures for Singular and Hypersingular Integrals
    N = length(x)
    # we evaluate the the Legendre polynomial up to order N-1
    # at the quadrature points
    M = evalLegendrePol(N,x)
    # evaluating the Legendre function of second kind using GSL
    # Q = Q_j(y)
    Q = repmat(1/(y-1)*ones(N,1),1,N )  - ((-1).^(repmat((0:N-1)',N,1 )))/(y+1) ;
    J = repmat((2*(0:N-1)'+1)/2,N,1 )
    W1 = sum(M.*Q.*J, 2)
    # evaluating the Legendre function of second kind using GSL


    # P_0(x_n) - P_1(x_n) *R_0(y)
    J1 = (M[:,1]-M[:,2]).*R_j(zeros(Int64,N,1),y*ones(N,1))
    # \sum_{j=1}^{N-2} (P_{j-1}(x_n) - P_{j+1}(x_n))*R_j(y)
    J2 = (M[:,1:N-2]-M[:,3:N]).*(R_j(repmat((1:N-2)',N,1), repmat(y*ones(N,1),1,N-2)) ) ;
    # P_{N-2}(x_n)*R_{N-1}(y)
    J3  = M[:,N-1].*R_j((N-1)*ones(Int64, N,1),y*ones(N,1))
    # P_{N-1}(x_n)*R_{N}(y)
    J4  = M[:,N].*R_j((N)*ones(Int64, N,1),y*ones(N,1))
    # adding everything up
    J = J1 + sum(J2,2) + J3 + J4;
    return  w.*J
end

function R_j(j::Array{Int64,2},y::Array{Float64,2})
    # function to numerically evaluate R_j(x) in Kolm, Rohklin paper
    return sf_legendre_Ql(j,y) + 0.25*log((y-1).^2)
end



function evalGeneralizedMoments(M::Int64,y::Float64,x::Array{Float64,1}, w::Array{Float64,1})
    Phi = evalPhi(M,y,x)
    M = Phi*diagm(w)
    # we are not using the modified quadratures in Kolm Rohklin
    m = sum(M,2)
end
