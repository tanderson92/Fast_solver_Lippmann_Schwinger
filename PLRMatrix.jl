
type PLRMat{T<:Number}
    # type for Partitioned Low Rank matrix
    blockType 
    childPLRMat::Array{PLRMat{T},2}
    # if the matrix is compressible we use a low rank matrix
    Ulowrank::LowRankMatrix{T}
    Udense::Array{T,2}
    # size of the matrix
    n::Int64
    m::Int64
    kMax::Int64 # maximum rank
    epsilon::Float64

    # default constructor
    PLRMat() = new()
end

function dense2PLR{T<:Number}(A::Array{T,2}, kMax::Int64 ; epsilon=1e-6)
    # creating a new node in the tree
    node = PLRMat{T}()
    # computing the size of A
    (n,m) = size(A)
    # filling some fields
    node.n    = n;
    node.m    = m;
    node.kMax = kMax;
    node.epsilon = epsilon;
    if n <= kMax || m <= kMax
        node.Udense = A
        node.blockType = 'd'
    else    
        # computing the randomized SVD
        (U,S,V) = randSVD(A, kMax+1)
        # testing the last eigenvalue
        if (S[end] < epsilon)
            # if it's true it means that the matrix is 
            # compressible
            node.Ulowrank = LowRankMatrix(U[:,1:kMax], spdiagm(S[1:kMax])*V[:,1:kMax]',n,m,kMax)
            node.blockType = 'c'
        else 
            # if the matrix is not compressible we need to build the
            # hierarchical structure 
            node.blockType = 'h'
            # creating the array with the PlR matrices
            node.childPLRMat = Array(PLRMat{T}, 4,4)
            # size of the blocks of the partition
            nlocal = floor(Integer, n/4);
            mlocal = floor(Integer, m/4);
            # indices for the new partition (in 16 blocks)
            indi = [0, nlocal, 2*nlocal, 3*nlocal, n];
            indj = [0, mlocal, 2*mlocal, 3*mlocal, m];
            for ii = 1:4
                for jj = 1:4
                    # matrix will be divide in 16 equally spaced 
                    indicesi = indi[ii]+1:indi[ii+1]
                    indicesj = indj[jj]+1:indj[jj+1]
                    node.childPLRMat[ii,jj] =  dense2PLR(A[indicesi,indicesj], kMax; epsilon=epsilon)
                end
            end
        end
    end

    return node
end

function matvecPLR{T}(H::PLRMat{T},b::Array{T,1})
    # check that they have the correct size
    @assert H.m == size(b)[1]
    # if the matrix is just a dense matrix
    if H.blockType == 'd'
        return H.Udense*b
    # if the matrix is a compressed matrix
    elseif H.blockType == 'c'
        return H.Ulowrank*b
    else 
        # otherwise just call it in a recursive fashion
        # defining the index parition
        nlocal = floor(Integer, H.n/4);
        mlocal = floor(Integer, H.m/4);
        indj = [0, mlocal, 2*mlocal, 3*mlocal, H.m];
        indi = [0, nlocal, 2*nlocal, 3*nlocal, H.n];
        y = zeros(H.n)[:]
        for ii = 1:4
            for jj = 1:4
                # the indices for each one of the applications
                indicesi = indi[ii]+1:indi[ii+1]
                indicesj = indj[jj]+1:indj[jj+1]
                y[indicesi] += matvecPLR(H.childPLRMat[ii,jj], b[indicesj]);
            end
        end
        return y
    end
end


import Base.*
function *{T}(M::PLRMat{T}, v::Array{T,1})
    @assert M.m == length(v) # make sure vector is right size for matrix
    return matvecPLR(M,v)
end
;
