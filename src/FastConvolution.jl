# File with the functions necessary to implement the
# sparsifiying preconditioner
# Ying 2014 Sparsifying preconditioners for the Lippmann-Schwinger Equation

include("Functions.jl")

using SpecialFunctions


struct FastM
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex{Float64},2}
    nu :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    # number of points in the original domain
    n  :: Int64
    m  :: Int64
    # frequency
    omega :: Float64
    quadRule :: String
    function FastM(GFFT,nu,ne,me,n,m,k; quadRule::String = "trapezoidal")
      return new(GFFT,nu,ne,me,n, m, k, quadRule)
    end
end

import Base.*

function Base.size(v::FastM, dim)
  size(v.nu, dim)
end

#function LinearAlgebra.mul!(Y::Vector{Complex{Float64}},M::FastM,b::Vector{Complex{Float64}},α::Bool,β::Bool)
#  Y = α * fastconvolution(M,b) + β*Y
#end

function LinearAlgebra.mul!(Y::AbstractArray{Complex{Float64},1},M::FastM,b::AbstractArray{Complex{Float64},1},α::Bool,β::Bool)
  Y = α * fastconvolution(M,b) + β*Y
end

function *(M::FastM, b::Array{Complex{Float64},1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT
    # dummy function to call fastconvolution
    return fastconvolution(M,b)
end




@inline function fastconvolution(M::FastM, b::AbstractArray{Complex{Float64},1})
    # function to overload the applyication of
    # M using a Toeplitz reduction via a FFT

    # computing G*(b nu)

    if M.quadRule == "trapezoidal"

      #obtaining the middle index
      indMiddle = round(Integer, M.n)

      # Allocate the space for the extended B
      BExt = zeros(Complex{Float64},M.ne, M.me);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.m]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fft(BExt)
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(BFft)

      # multiplication by omega^2
      B = M.omega^2*(BExt[M.n:M.n+M.n-1, M.m:M.m+M.m-1]);

    elseif M.quadRule == "Greengard_Vico"
      # for this we use the Greengard Vico method in the
      # frequency domain

      # Allocate the space for the extended B
      BExt = zeros(Complex{Float64},M.ne, M.me);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.m]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fftshift(fft(BExt))
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(ifftshift(BFft))

      # multiplication by omega^2
      B = M.omega^2*(BExt[1:M.n, 1:M.m]);

    end

    # returning b + G*(b nu)
    return (b + B[:])
end

@inline function FFTconvolution(M::FastM, b::Array{Complex{Float64},1})
    # function to overload the applyication of
    # convolution of b times G

    if M.quadRule == "trapezoidal"

      #obtaining the middle index
      indMiddle = round(Integer, M.n)

      # Allocate the space for the extended B
      BExt = zeros(Complex{Float64},M.ne, M.ne);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.m]= reshape(M.nu.*b,M.n,M.m) ;

      # Fourier Transform
      BFft = fft(BExt)
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(BFft)

      # multiplication by omega^2
      B = (BExt[indMiddle: indMiddle+M.n-1, indMiddle:indMiddle+M.n-1]);

    elseif M.quadRule == "Greengard_Vico"
      # for this we use the Greengard Vico method in the
      # frequency domain

      # Allocate the space for the extended B
      BExt = zeros(Complex{Float64},M.ne, M.ne);
      # Apply spadiagm(nu) and ented by zeros
      BExt[1:M.n,1:M.n]= reshape(b,M.n,M.n) ;

      # Fourier Transform
      BFft = fftshift(fft(BExt))
      # Component-wise multiplication
      BFft = M.GFFT.*BFft
      # Inverse Fourier Transform
      BExt = ifft(ifftshift(BFft))

      # multiplication by omega^2
      B = (BExt[1:M.n, 1:M.n]);
    end
    return  B[:]
end

# # #this is the sequantial version for sampling G
# function sampleG(k,X,Y,indS, D0)
#     # function to sample the Green's function at frequency k
#     Gc = zeros(Complex{Float64}, length(indS), length(X))
#     for i = 1:length(indS)
#         ii = indS[i]
#         r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
#         r[ii] = 1;
#         Gc[i,:] = 1im/4*hankelh1(0, k*r)*h^2;
#         Gc[i,ii]= 1im/4*D0*h^2;
#     end
#     return Gc
# end

function buildFastConvolution(x::Array{Float64,1},y::Array{Float64,1},
                              h::Float64,k,nu::Function; quadRule::String = "trapezoidal")

  if quadRule == "trapezoidal"

    (ppw,D) = referenceValsTrapRule();
    D0      = D[round(Int,k*h)];
    (n,m) = length(x), length(y)
    Ge    = buildGConv(x,y,h,n,m,D0,k);
    GFFT  = fft(Ge);
    X = repeat(x, 1, m)[:]
    Y = repeat(y', n,1)[:]

    return FastM(GFFT,nu(X,Y),2*n-1,2*m-1,n, m, k);

  elseif quadRule == "Greengard_Vico"

      Lp = 4*(abs(x[end] - x[1]) + h)
      L  =   (abs(x[end] - x[1]) + h)*1.5
      (n,m) = length(x), length(y)
      X = repeat(x, 1, m)[:]
      Y = repeat(y', n,1)[:]

      # this is depending if n is odd or not
      if mod(n,2) == 0
        kx = (-(2*n):1:(2*n-1));
        ky = (-(2*m):1:(2*m-1));

        KX = (2*pi/Lp)*repeat(kx, 1, 4*m);
        KY = (2*pi/Lp)*repeat(ky', 4*n,1);

        S = sqrt(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)
        return FastM(GFFT, nu(X,Y), 4*n, 4*m,
                     n, m, k , quadRule="Greengard_Vico");
      else
        # kx = (-2*(n-1):1:2*(n-1) )/4;
        # ky = (-2*(m-1):1:2*(m-1) )/4;

        # KX = (2*pi/Lp)*repeat(kx, 1, 4*m-3);
        # KY = (2*pi/Lp)*repeat(ky', 4*n-3,1);

        # S = sqrt(KX.^2 + KY.^2);

        # GFFT = Gtruncated2D(L, k, S)

        # return FastM(GFFT,nu(X,Y),4*n-3,4*m-3,
        #              n,m, k,quadRule = "Greengard_Vico");

        kx = (-2*n:1:2*n-1);
        ky = (-2*m:1:2*m-1);

        KX = (2*pi/Lp)*repeat( kx, 1,4*m);
        KY = (2*pi/Lp)*repeat(ky',4*n,  1);

        S = sqrt(KX.^2 + KY.^2);

        GFFT = Gtruncated2D(L, k, S)

        return FastM(GFFT,nu(X,Y),4*n,4*m,
                     n,m, k,quadRule = "Greengard_Vico");


    end
  end
end


function sampleG(k,X,Y,indS, D0)
    # function to sample the Green's function at frequency k

  #   R  = SharedArray(Float64, length(indS), length(X))
  #   Xshared = SharedArray(X)
  #   Yshared = SharedArray(Y)
  #   @sync begin
  #     @parallel for i = 1:length(indS)
  #   #for i = 1:length(indS)
  #     ii = indS[i]
  #     R[i,:]  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
  #     R[i,ii] = 1;
  #   end
  # end

  h = abs(X[2] - X[1])

  R  = SharedArray{Float64}(length(indS), length(X))
  Xshared = convert(SharedArray, X)
  Yshared = convert(SharedArray, Y)
  begin
    for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,:]  = sqrt.( (Xshared.-Xshared[ii]).^2 + (Yshared.-Yshared[ii]).^2);
      R[i,ii] = 1;
    end
  end

    #Gc = (h^2*1im/4)*hankelh1(0, k*R)*h^2;
    Gc = sampleGkernelpar(k,R,h);
    for i = 1:length(indS)
        ii = indS[i]
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end


function sampleGConv(k,X,Y,indS, fastconv::FastM)
    # function to sample the Green's function at frequency k
    # using convolution

  #   R  = SharedArray(Float64, length(indS), length(X))
  #   Xshared = SharedArray(X)
  #   Yshared = SharedArray(Y)
  #   @sync begin
  #     @parallel for i = 1:length(indS)
  #   #for i = 1:length(indS)
  #     ii = indS[i]
  #     R[i,:]  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
  #     R[i,ii] = 1;
  #   end
  # end

  R  = zeros(Complex{Float64}, length(indS), length(X))
   for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,ii] = 1;
    end

   Gc =  zeros(Complex{Float64}, length(indS), length(X))
    for i = 1:length(indS)
        Gc[i,:]= FFTconvolution(fastconv, R[i,:][:])
    end
    return Gc
end



## TODO: parallelize the sampling of the 3D Green's function
# @everywhere function sampleG3DParallel(k,X,Y,Z, indS, D0)
#   # function to sample the Green's function at frequency k

#   R  = SharedArray(Float64, length(indS), length(X))
#   Xshared = convert(SharedArray, X)
#   Yshared = convert(SharedArray, Y)
#   Zshared = convert(SharedArray, Z)
#   @sync begin
#       for i = 1:length(indS)
#       #for i = 1:length(indS)
#       ii = indS[i]
#       R[i,:]  = sqrt( (Xshared-Xshared[ii]).^2 + (Yshared-Yshared[ii]).^2 + (Zshared-Zshared[ii]).^2);
#       R[i,ii] = 1;
#     end
#   end

#     Gc = (exp(1im*k*R)*h^2)./(4*pi*R) ;
#     for i = 1:length(indS)
#         ii = indS[i]
#         Gc[i,ii]= 1im/4*D0*h^2;
#     end
#     return Gc
# end

## Parallel functions to sample the Kernel

function myrange(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0, 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in LinRange(0,size(q,2),nchunks+1)]
    1:size(q,1), splits[idx]+1:splits[idx+1]
end

function sampleGkernelpar(k,r::Array{Float64,1},h)
  n  = length(r)
  println("Sample kernel parallel loop ")
  G = SharedArray(Complex{Float64},n)
  rshared = convert(SharedArray, r)
  for ii = 1:n
          @inbounds  G[ii] = 1im/4*hankelh1(0, k*rshared[ii])*h^2;
  end
  return sdata(G)
end

## two different versions of the same function with slight different input

function sampleGkernelpar(k,R::Array{Float64,2},h)
  (m,n)  = size(R)
  println("Sample kernel parallel loop with chunks ")
  G = SharedArray(Complex{Float64},m,n)
  @time Rshared = convert(SharedArray, R)
  @sync begin
        for p in procs(G)
             @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, Rshared,k,h)
        end
    end
  return sdata(G)
end

@everywhere function sampleGkernelpar(k,R::SharedArray{Float64,2},h)
  (m,n)  = size(R)
  #println("Sample kernel parallel loop with chunks 2 ")
  G = SharedArray{Complex{Float64}}(m,n)
  @sync begin
        for p in procs(G)
            @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, R,k,h)
        end
    end
  return sdata(G)
end


# little convenience wrapper
@everywhere sampleGkernel_shared_chunk!(q,u,k,h) = sampleGkernel_chunk!(q,u,k,h, myrange(q)...)

@everywhere @inline function sampleGkernel_chunk!(G, R,k::Float64,h::Float64,
                                                  irange::UnitRange{Int64}, jrange::UnitRange{Int64})
    #@show (irange, jrange)  # display so we can see what's happening
    # println(myid())
    # println(typeof(irange))
    alpha = 1im/4*h^2
    for i in irange
      for j in jrange
        @inbounds G[i,j] = alpha*hankelh1(0, k*R[i,j]);
      end
    end
end

##################################################################################
### Routines to sample compute the Sparsified version of the matrices


function referenceValsTrapRule()
    # modification to the Trapezoidal rule for logarithmic singularity
    # as explained in
    # R. Duan and V. Rokhlin, High-order quadratures for the solution of scattering problems in
    # two dimensions, J. Comput. Phys.,
    x = 2.0.^(-(0:5))[:]
    w = [1-0.892*im, 1-1.35*im, 1-1.79*im, 1- 2.23*im, 1-2.67*im, 1-3.11*im]
    return (x,w)
end



# @inline function indexMatrix(indI::Int64, indJ::Int64, indK::Int64, n::Int64,m::Int64,k::Int64)
#   # function easily compute the global index from dimensional ones
#   # for a third order tensor

# end

function buildGConv(x,y,h::Float64,n::Int64,m::Int64,D0,k::Float64)
    # function to build the convolution vector for the
    # fast application of the convolution

    # this is built for odd n and odd m.

    if mod(n,2) == 1
      # build extended domain
      xe = collect((x[1]-(n-1)/2*h):h:(x[end]+(n-1)/2*h));
      ye = collect((y[1]-(m-1)/2*h):h:(y[end]+(m-1)/2*h));

      Xe = repeat(xe, 1, 2*m-1);
      Ye = repeat(ye', 2*n-1,1);


    else

      println("so far only works for n odd")
      # to be done
      # # build extended domain
      # xe = collect((x[1]-n/2*h):h:(x[end]+n/2*h));
      # ye = collect((y[1]-m/2*h):h:(y[end]+m/2*h));

      # Xe = repeat(xe, 1, 2*m-1);
      # Ye = repeat(ye', 2*n-1,1);
      # # to avoid evaluating at the singularity
      # indMiddle = m

    end

    R = sqrt(Xe.^2 + Ye.^2);

    # to avoid evaluating at the singularity
    indMiddle = find(R.==0)[1]    # we modify R to remove the zero (so we don't )
    R[indMiddle] = 1;
    # sampling the Green's function
    Ge = sampleGkernelpar(k,R,h)
    #Ge = pmap( x->1im/4*hankelh1(0,k*x)*h^2, R)
    # modiyfin the diagonal with the quadrature
    # modification
    Ge[indMiddle] = 1im/4*D0*h^2;

    return Ge

end


function buildGConvPar(x,y,h,n,m,D0,k)

    # build extended domain
    xe = collect((x[1]-(n-1)/2*h):h:(x[end]+(n-1)/2*h));
    ye = collect((y[1]-(m-1)/2*h):h:(y[end]+(m-1)/2*h));

    Xe = repeat(xe, 1, 2*m-1);
    Ye = repeat(ye', 2*n-1,1);

    R = sqrt(Xe.^2 + Ye.^2);
    # to avoid evaluating at the singularity
    indMiddle = round(Integer, m)
    # we modify R to remove the zero (so we don't )
    R[indMiddle,indMiddle] = 1;
    # sampling the Green's function
    Ge = sampleGkernelpar(k,R,h)
    #Ge = pmap( x->1im/4*hankelh1(0,k*x)*h^2, R)
    # modiyfin the diagonal with the quadrature
    # modification
    Ge[indMiddle,indMiddle] = 1im/4*D0*h^2;

return Ge

end

function buildConvMatrix(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},D0::Complex{Float64}, h::Float64)
    # function to build the convolution matrix
    @assert length(X) == length(Y)
    N = length(X);

    G = zeros(Complex{Float64}, N, N);

    r = zeros(Float64,N)
    for ii = 1:N
            r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
            r[ii] = 1;
            G[ii,:] =  1im/4*hankelh1(0, k*r)*h^2;
            G[ii,ii]=  1im/4*D0*h^2;
    end

    return G
end




######################## Randomized methods ##############################
######not working

# function entriesSparseARand(k,X,Y,D0, n ,m)
#   # we need to have an even number of points
#   # We compute the entries for the matrix A using randomized methods
#   @assert mod(length(X),2) == 1
#   Entries  = Array{Complex{Float64}}[]
#   Indices  = Array{Int64}[]

#   N = n*m;

#   # computing the entries for the interior
#   indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
#   indVolC = setdiff(collect(1:N),indVol);
#   GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);

#   # for  x = xmin, y = 0
#   indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
#   indC = setdiff(collect(1:N),indFz1);
#   GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices, [0,1,n,n+1,-n, -n+1]); #'

#   # for  x = xmax, y = 0
#   indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
#   indC = setdiff(collect(1:N),indFz2);
#   GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,n,n-1,-n, -n-1]); #'

#   # for  y = ymin, x = 0
#   indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
#   indC = setdiff(collect(1:N),indFx1);
#   GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,n,n+1, n-1]); #'

#   # for  y = ymin, x = 0
#   indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
#   indC = setdiff(collect(1:N),indFx2);
#   GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[-1,0,1,-n,-n+1, -n-1]); #'

#   # For the corners
#   indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
#   indcorner2 = round(Integer, n + [0,-1, n,n-1]);
#   indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
#   indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

#   indC = setdiff(collect(1:N),indcorner1);
#   GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
#   # computing the sparsifying correction
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,1, n,n+1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner2);
#   GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
#   # computing the sparsifying correction
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,-1, n,n-1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner3);
#   GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,1, -n,-n+1]); #'

#   #'
#   indC = setdiff(collect(1:N),indcorner4);
#   GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indC ];
#   (n,l) = size(GSampled);
#   GsubSampled = GSampled*(randn(l,20*length(indVol)) + 1im*randn(l,20*length(indVol)));


#   # computing the sparsifying correction
#   (U,s,V) = svd(GsubSampled);
#   push!(Entries,U[:,end]'); #'
#   push!(Indices,[0,-1, -n,-n-1]); #'

#   return (Indices, Entries)
# end
