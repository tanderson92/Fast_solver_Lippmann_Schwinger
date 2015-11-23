# type FastConv
#     GFFT :: Array{Complex128,2}
#     nf :: Int64
#     n  :: Int64
# end

using Devectorize


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
	  #obtaining the middle index
    indMiddle = round(Integer, M.n-1 + (M.n+1)/2)
	  # extended b
    BExt = zeros(Complex128,M.ne, M.ne);
   	BExt[1:M.n,1:M.n]= reshape(M.nu.*b,M.n,M.n) ;

   	# Fourier Transform
   	BFft = fft(BExt)
   	# Component-wise multiplication
   	BFft = M.GFFT.*BFft
   	# Inverse Fourier Transform
   	BExt = ifft(BFft)

    # multiplication by omega^2
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

function entriesSparseA(k,X,Y,D0, n ,m)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]
  Indices  = Array{Int64}[]

  N = n*m;

  # computing the entries for the interior
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + [-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,n,n-1,n+1,-n,-n-1, -n+1]);

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, [0,1,n,n+1,-n, -n+1]); #'

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(n-1)/2 + [-1,0,n,n-1,-n, -n-1]);
  indC = setdiff(collect(1:N),indFz2);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,n,n-1,-n, -n-1]); #'

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,n,n+1, n-1]); #'

  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
  indC = setdiff(collect(1:N),indFx2);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[-1,0,1,-n,-n+1, -n-1]); #'

  # For the corners
  indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
  indcorner2 = round(Integer, n + [0,-1, n,n-1]);
  indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
  indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,1, n,n+1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,-1, n,n-1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,1, -n,-n+1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,-1, -n,-n-1]); #'

  return (Indices, Entries)
end


function buildSparseA(k,X,Y,D0, n ,m)
# function that build the sparsigying preconditioner

    Ind = reshape(collect(1:n*m),n,m);

    (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);


    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);

    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);


    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end


function createIndices(row, col, val)

  @assert length(col) == length(val)
  nn = length(col);
  mm = length(row);

  Row = kron(row, ones(Int64, nn));
  Col = kron(ones(Int64,mm), col) + Row;
  Val = kron(ones(Int64,mm), val)
  return (Row,Col,Val)
end


function buildConvMatrix(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},D0::Complex128, h::Float64)
    # function to build the convolution matrix
    @assert length(X) == length(Y)
    N = length(X);

    G = zeros(Complex128, N, N);

    r = zeros(Float64,N)
    for ii = 1:N
            r  = sqrt( (X-X[ii]).^2 + (Y-Y[ii]).^2);
            r[ii] = 1;
            G[ii,:] =  1im/4*hankelh1(0, k*r)*h^2;
            G[ii,ii]=  1im/4*D0*h^2;
    end

    return G
end


