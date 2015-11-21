abstract Matrix

type LowRankMatrix{T<:Number} <: Matrix
    ## a rank-1 matrix type
    # A = u*v'
    # m is size of matrix
    U::Array{T, 2}
    V::Array{T, 2}
    n::Int64
    m::Int64
    k::Int64
end

function LowRankMatrix{T<:Number}(U::Array{T, 2}, V::Array{T, 2})
    @assert size(U)[2] == size(V)[1] # we're only dealing with square matrices
    return LowRankMatrix(U, V, size(U)[1], size(V)[2], size(V)[1])
end

function randSVD{T<:Number}(A::Array{T,2}, k::Int64, oversample = 5)
    # function to compute the randomized SVD of the matrix A
    # we hope to extract only the first k biggest eigenvalues
    (n,m) = size(A)
    #number of random vectors
    nRandVecs = min(k+oversample,min(n,m))
    randVecs = randn(nRandVecs, n);
    # we multiply A by the random vectors
    R = randVecs*A;
    # we compute the row space of A using a skinny svd
    Q = svd(R')[1]
    P = A*Q;
    (U, S, W) = svd(P)
    V = Q*W
    return (U[:, 1:k], S[1:k], V[:,1:k])
end


function compressLowRankMatrix{T<:Number}(A::Array{T, 2}, epsilon::Float64 )
    # function to compress into a low rank matrix
    (U, S,V) = svd(A)
    # computing the epsilon rank
    epsRank = length(find(S.>0.1));
    return LowRankMatrix(U[:,1:epsRank], spdiagm(S[1:epsRank])*(V[:,1:epsRank])',
                         size(A)[1], size(A)[2], epsRank)
end

import Base.*
function *{T<:Number}(M::LowRankMatrix{T}, v::Array{T,1})
    # Matrix vector multiplication overload
    @assert M.m == length(v) # make sure vector is right size for matrix
    c = M.V*v
    return M.U*c
end
function *{T<:Number}(M::LowRankMatrix{T}, N::Array{T,2})
    # Matrix-Matrix product overload 
    @assert M.m == size(N)[1] # make sure vector is right size for matrix
    C = M.V*N
    return M.U*C
end

;
