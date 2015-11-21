# type FastConv
#     GFFT :: Array{Complex128,2}
#     nf :: Int64
#     n  :: Int64
# end

type FastConv{T}
    GFFT :: Array{T,2}
    nf :: Int64
    n  :: Int64
end


type FastM
    # type to encapsulate the fast application of M = I + omega^2G*spadiagm(nu)
    GFFT :: Array{Complex128,2}
    nu :: Array{Float64,1}
    # number of points in the extended domain
    ne :: Int64
    me :: Int64
    n  :: Int64
    m  :: Int64
    omega :: Float64
end


import Base.*

function *(M::FastM, b::Array{Complex128,1})
	indMiddle = round(Integer, M.n-1 + (M.n+1)/2)
	# extended b
    BExt = zeros(Complex128,M.ne, M.ne);
   	BExt[1:M.n,1:M.n]= reshape(M.nu.*b,M.n,M.n) ;
   	#BExt[1:M.n,1:M.n]= reshape(b,M.n,M.n) ;
   	# Fourier Transform
   	BFft = fft(BExt)
   	# Component-wise multiplication
   	BFft = M.GFFT.*BFft
   	# Inverse Fourier Transform
   	BExt = ifft(BFft)
   	B = M.omega^2*(BExt[indMiddle: indMiddle+M.n-1, indMiddle:indMiddle+M.n-1]);
   	return (b + B[:])
end



function sampleG(k,X,Y,indS, D0)
    # function to sample the Green's function at frequency k
    Gc = zeros(Complex128, length(indS), length(X))
    for i = 1:length(indS)
        ii = indS[i]
        r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
        r[ii] = 1;
        Gc[i,:] =  1im/4*hankelh1(0, k*r)*h^2;
        Gc[i,ii]= 1im/4*D0*h^2;
    end
    return Gc
end
