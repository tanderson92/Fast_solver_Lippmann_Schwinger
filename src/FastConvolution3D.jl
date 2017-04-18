# File with the function necessary to implement the fast convolution
# in 3D and all the necessary machinery to build the preconditioner

include("FastConvolution.jl")


type FastM3D
    ## we may want to add an FFT plan to make the evaluation faster
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex128,3}
    nu :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    le :: Int64
    # number of points in the original domain
    n  :: Int64
    m  :: Int64
    l  :: Int64
    # frequency
    omega :: Float64
    quadRule :: String
    function FastM3D(GFFT,nu,ne,me,le,n,m,l,k;quadRule::String = "Greengard_Vico")
      return new(GFFT,nu,ne,me,le,n,m,l,k,quadRule)
    end
end

import Base.*


function *(M::FastM3D, b::Array{Complex128,1}; verbose::Bool=false)
    # multiply by nu, compute the convolution and then
    # multiply by omega^2
    B = M.omega^2*(FFTconvolution(M,M.nu.*b, verbose=verbose))

    return (b + B)
end

@inline function FFTconvolution(M::FastM3D, b::Array{Complex128,1};
                                verbose::Bool=false )
    # function to compute the convolution with the convolution kernel
    # defined within the FastM3D type using the FFT
    # TODO add a fft plan in here to accelerate the speed
    # input: M::FastM3D type containing the convolution kernel
    #        b::Array{Complex128,1} vector to apply the conv kernel
    verbose && println("Application of the 3D convolution")
    # Allocate the space for the extended B
    BExt = zeros(Complex128,M.ne, M.ne, M.le);
    # zero padding
    BExt[1:M.n,1:M.m,1:M.l]= reshape(b,M.n,M.m,M.l) ;

    # Fourier Transform
    BFft = fftshift(fft(BExt))
    # Component-wise multiplication
    BFft = M.GFFT.*BFft
    # Inverse Fourier Transform
    BExt = ifft(ifftshift(BFft))

    # multiplication by omega^2
    B = BExt[1:M.n,1:M.m,1:M.l];

    return B[:]
end



# we need to write the convolution in 3D, the aim is to have 2 and 3 convolution
function buildFastConvolution3D(x,y,z,X,Y,Z,h,k,nu; quadRule::String = "Greengard_Vico")

 if quadRule == "Greengard_Vico"

      Lp = 4*(abs(x[end] - x[1]) + h)
      L = (abs(x[end] - x[1]) + h)*1.8
      (n,m,l) = length(x), length(y), length(z)
      #LPhysical = abs(x[end]-x[1])+h;

      if mod(n,2) == 0

        kx = (2*pi/Lp)*collect(-(2*n):1:(2*n-1));
        ky = (2*pi/Lp)*collect(-(2*m):1:(2*m-1));
        kz = (2*pi/Lp)*collect(-(2*l):1:(2*l-1));

        # KX = 2*pi*[ kx[i] + 0*j   + 0*p   for i=1:4*n, j=1:4*m, p=1:4*l ]
        # KY = 2*pi*[ 0*i   + ky[j] + 0*p   for i=1:4*n, j=1:4*m, p=1:4*l ]
        # KZ = 2*pi*[ 0*i   + 0*j   + kz[p] for i=1:4*n, j=1:4*m, p=1:4*l ]

        # S = sqrt(KX.^2 + KY.^2 + KZ.^2);
        ## S  = [ sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2) for i=1:4*n, j=1:4*m, p=1:4*l ]

        # GFFT = Gtruncated3D(L, k, S)

        ### GFFT = [ Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2)) for i=1:4*n, j=1:4*m, p=1:4*l]

        # Computing the convolution kernel (we just use a for loop in order to save memory)
        GFFT = zeros(Complex128, 4*n, 4*m, 4*l)

        for ii=1:4*n, jj=1:4*m, pp=1:4*l
          GFFT[ii,jj,pp]  = Gtruncated3D(L,k,sqrt(kx[ii]^2 + ky[jj]^2 + kz[pp]^2));
        end

        return FastM3D(GFFT,nu(X,Y,Z),4*n,4*m,4*l, n, m,l, k ,quadRule = "Greengard_Vico");
      else

        kx = 2*pi*collect(-2*(n-1):1:2*(n-1) )/4;
        ky = 2*pi*collect(-2*(m-1):1:2*(m-1) )/4;
        kz = 2*pi*collect(-2*(m-1):1:2*(m-1) )/4;

        # KX = [ kx[i] + 0*j   + 0*p   for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]
        # KY = [ 0*i   + ky[j] + 0*p   for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]
        # KZ = [ 0*i   + 0*j   + kz[p] for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # S = sqrt(KX.^2 + KY.^2 + KZ.^2);

        ## S  = [ sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2) for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # GFFT = Gtruncated3D(L, k, S)

        ### GFFT = [ Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2)) for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3 ]

        # Computing the convolution kernel (we just use a for loop in order to save memory)
        GFFT = zeros(Complex128, 4*n-3, 4*m-3, 4*l-3)

        # This loop can be easily parallelized
        for i=1:4*n-3, j=1:4*m-3, p=1:4*l-3
          GFFT[i,j,p]  = Gtruncated3D(L,k,sqrt(kx[i]^2 + ky[j]^2 + kz[p]^2));
        end

        return FastM3D(GFFT,nu(X,Y,Z),4*n-3,4*m-3,4*l-3,n, m, l,k,quadRule = "Greengard_Vico");

    end
  end
end



@everywhere function sampleG3D(k,X,Y,Z, indS, fastconv::FastM3D)
  # function to sample the 3D Green's function in the nodes given by indS
  # input:    k: float64 frequency
  #           X: mesh contaning the X position of each point
  #           Y: mesh contaning the Y position of each point
  #           Z: mesh contaning the Y position of each point
  #           indS: indices in which the sources are located
  #           fastconv: FastM3D type for the application of the
  #                     discrete convolution kernel

  R  = zeros(Complex128, length(indS), length(X))
   for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,ii] = 1;
    end

   Gc =  zeros(Complex128, length(indS), length(X))
   # this can be parallelized but then, we may have cache
   # aceess problem
    for i = 1:length(indS)
        Gc[i,:]= FFTconvolution(fastconv, R[i,:][:])
    end
    return Gc
end

@everywhere function sampleG3D(k,X,Y,Z, indS, D0)
  # function to sample the 3D Green's function in the nodes given by indS
  # input:    k: float64 frequency
  #           X: mesh contaning the X position of each point
  #           Y: mesh contaning the Y position of each point
  #           Z: mesh contaning the Y position of each point
  #           indS: indices in which the sources are located
  #           D0: diagonal modification based on the kh, this is given as a
  #               parameter but it should be correctly handled in newer versions
  #               of the code.
  # function to sample the Green's function at frequency k

  R  = SharedArray(Float64, length(indS), length(X))
  Xshared = convert(SharedArray, X)
  Yshared = convert(SharedArray, Y)
  Zshared = convert(SharedArray, Z)
  @sync begin
    @parallel for i = 1:length(indS)
      #for i = 1:length(indS)
      ii = indS[i]
      R[i,:]  = sqrt( (Xshared-Xshared[ii]).^2 + (Yshared-Yshared[ii]).^2 + (Zshared-Zshared[ii]).^2);
      R[i,ii] = 1;
    end
  end

  # sampling the Green's function in the given points
    Gc = (exp(1im*k*R)*h^2)./(4*pi*R) ;
    for i = 1:length(indS)
        ii = indS[i]
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end

#######################################################
### function to enhance parallelism
##################################################
# function SampleGtruncated3D(GFFT, L,k,kx,ky,kz)
#     n = length(kx)
#     m = length(ly)
#     l = length(kz)

#     R  = SharedArray(Float64, length(indS), length(X))
#     kxshared = convert(SharedArray, kx)
#     kyshared = convert(SharedArray, ky)
#     kzshared = convert(SharedArray, kz)

# end




# @everywhere function myrange(q::SharedArray)
#     idx = indexpids(q)
#     if idx == 0
#         # This worker is not assigned a piece
#         return 1:0, 1:0
#     end
#     nchunks = length(procs(q))
#     splits = [round(Int, s) for s in linspace(0,size(q,2),nchunks+1)]
#     1:size(q,1), splits[idx]+1:splits[idx+1]
# end

# @everywhere function sampleGkernelparTruncated3D(k,r::Array{Float64,1},h)
#   n  = length(r)
#   println("Sample kernel parallel loop ")
#   G = SharedArray(Complex128,n)
#   rshared = convert(SharedArray, r)
#   @sync @parallel for ii = 1:n
#           @inbounds  G[ii] = 1im/4*hankelh1(0, k*rshared[ii])*h^2;
#   end
#   return sdata(G)
# end

# ## two different versions of the same function with slight different input

# @everywhere function sampleGkernelpar(k,R::Array{Float64,2},h)
#   (m,n)  = size(R)
#   println("Sample kernel parallel loop with chunks ")
#   G = SharedArray(Complex128,m,n)
#   @time Rshared = convert(SharedArray, R)
#   @sync begin
#         for p in procs(G)
#              @async remotecall_fetch(sampleGkernel_shared_chunk!,p, G, Rshared,k,h)
#         end
#     end
#   return sdata(G)
# end


# # little convenience wrapper
# @everywhere sampleGkernel_shared_chunk!(q,u,k,h) = sampleGkernel_chunk!(q,u,k,h, myrange(q)...)

# @everywhere @inline function sampleGkernel_chunk!(G,R,k::Float64,h::Float64,
#                                                   irange::UnitRange{Int64}, jrange::UnitRange{Int64})
#     #@show (irange, jrange)  # display so we can see what's happening
#     # println(myid())
#     # println(typeof(irange))
#     alpha = 1im/4*h^2
#     for i in irange
#       for j in jrange
#         @inbounds G[i,j] = alpha*hankelh1(0, k*R[i,j]);
#       end
#     end
# end

