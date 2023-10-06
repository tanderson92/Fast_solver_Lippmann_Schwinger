## File with the files for the construction of the Sparsifying Matrices 

include("FastConvolution.jl")

function entriesSparseA(k,X,Y,D0, n ,m)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex{Float64}}[]
  Indices  = Array{Int64}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  # extracting indices at the center of the stencil
  indVol = round.(Integer, n*(m-1)/2 .+ (n+1)/2 .+ IndRelative[:] );
  # extracting only the far field indices
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVolC];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:]);

  # this is for the edges

  # for  x = xmin, y = 0
  indFz1 = round.(Integer, n*(m-1)/2 .+1 .+ IndRelative[:,2:3][:]);
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[:,2:3][:]); #'

  # for  x = xmax, y = 0
  indFz2 = round.(Integer, n*(m-1)/2 .+ IndRelative[:,1:2][:]);
  indC = setdiff(collect(1:N),indFz2);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:,1:2][:]); #'

  # for  y = ymin, x = 0
  indFx1 = round.(Integer, (n+1)/2 .+ IndRelative[2:3,:][:]);
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,:][:]); #'

  # for  y = ymin, x = 0
  indFx2 = round.(Integer, N - (n+1)/2 .+ IndRelative[1:2,:][:]);
  indC = setdiff(collect(1:N),indFx2);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[1:2,:][:]); #'

  # For the corners
  indcorner1 = round.(Integer, 1       .+ IndRelative[2:3,2:3][:]);
  indcorner2 = round.(Integer, n       .+ IndRelative[2:3,1:2][:]);
  indcorner3 = round.(Integer, n*m-n+1 .+ [0, 1,-n,-n+1]);
  indcorner4 = round.(Integer, n*m     .+ [0,-1,-n,-n-1]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries, U[:,end]'); #'
  push!(Indices, IndRelative[2:3,1:2][:]); #'

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

function entriesSparseAConv(k,X,Y,fastconv::FastM, n ,m)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex{Float64}}[]
  Indices  = Array{Int64}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  # extracting indices at the center of the stencil
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + IndRelative[:] );
  # extracting only the far field indices
  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleGConv(k,X,Y,indVol, fastconv)[:,indVolC];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:]);

  # this is for the edges

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + IndRelative[:,2:3][:]);
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleGConv(k,X,Y,indFz1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[:,2:3][:]); #'

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(m-1)/2 + IndRelative[:,1:2][:]);
  indC = setdiff(collect(1:N),indFz2);
  GSampled = sampleGConv(k,X,Y,indFz2, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[:,1:2][:]); #'

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + IndRelative[2:3,:][:]);
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleGConv(k,X,Y,indFx1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,:][:]); #'

  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + IndRelative[1:2,:][:]);
  indC = setdiff(collect(1:N),indFx2);
  GSampled = sampleGConv(k,X,Y,indFx2, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,IndRelative[1:2,:][:]); #'

  # For the corners
  indcorner1 = round(Integer, 1       + IndRelative[2:3,2:3][:]);
  indcorner2 = round(Integer, n       + IndRelative[2:3,1:2][:]);
  indcorner3 = round(Integer, n*m-n+1 + [0, 1,-n,-n+1]);
  indcorner4 = round(Integer, n*m     + [0,-1,-n,-n-1]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleGConv(k,X,Y,indcorner1, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices, IndRelative[2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleGConv(k,X,Y,indcorner2, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries, U[:,end]'); #'
  push!(Indices, IndRelative[2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleGConv(k,X,Y,indcorner3, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,1, -n,-n+1]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleGConv(k,X,Y,indcorner4, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,[0,-1, -n,-n-1]); #'

  return (Indices, Entries)
end



function entriesSparseG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                        D0::Complex{Float64}, n::Int64 ,m::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex{Float64}}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  indVol = round.(Integer, n*(m-1)/2 + (n+1)/2 .+ IndRelative[:] );
  GSampled = sampleG(k,X,Y,indVol, D0)[:,indVol];

  push!(Entries,GSampled);

  # for  x = xmin, y = 0
  indFz1 = round.(Integer, n*(m-1)/2 +1 .+ [0,1,n,n+1,-n, -n+1]);
  GSampled = sampleG(k,X,Y,indFz1, D0)[:,indFz1];

  push!(Entries,GSampled);

  # for  x = xmax, y = 0
  indFz2 = round.(Integer, n*(m-1)/2 .+ [-1,0,n,n-1,-n, -n-1]);
  GSampled = sampleG(k,X,Y,indFz2, D0)[:,indFz2];

  push!(Entries,GSampled);

  # for  y = ymin, x = 0
  indFx1 = round.(Integer, (n+1)/2 .+ [-1,0,1,n,n+1, n-1]);
  GSampled = sampleG(k,X,Y,indFx1, D0)[:,indFx1];

  push!(Entries,GSampled);


  # for  y = ymin, x = 0
  indFx2 = round.(Integer, N - (n+1)/2 .+ [-1,0,1,-n,-n+1, -n-1]);
  GSampled = sampleG(k,X,Y,indFx2, D0)[:,indFx2];

  push!(Entries,GSampled);

  # For the corners
  indcorner1 = round.(Integer, 1 .+ [0,1, n,n+1]);
  indcorner2 = round.(Integer, n .+ [0,-1, n,n-1]);
  indcorner3 = round.(Integer, n*m-n+1 .+ [0,1, -n,-n+1]);
  indcorner4 = round.(Integer, n*m .+ [0,-1, -n,-n-1]);

  GSampled = sampleG(k,X,Y,indcorner1, D0)[:,indcorner1];
  push!(Entries,GSampled);


  #'
  GSampled = sampleG(k,X,Y,indcorner2, D0)[:,indcorner2];
  push!(Entries,GSampled);

  #'
  GSampled = sampleG(k,X,Y,indcorner3, D0)[:,indcorner3];
  push!(Entries,GSampled);

  #'
  GSampled = sampleG(k,X,Y,indcorner4, D0)[:,indcorner4];
  push!(Entries,GSampled);

  return Entries
end


function entriesSparseGConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       fastconv::FastM, n::Int64 ,m::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  @assert mod(length(X),2) == 1
  Entries  = Array{Complex{Float64}}[]

  IndRelative = zeros(Int64,3,3)
  IndRelative = [-n-1 -n -n+1;
                   -1  0    1;
                  n-1  n  n+1]

  N = n*m;

  # computing the entries for the interior
  indVol = round(Integer, n*(m-1)/2 + (n+1)/2 + IndRelative[:] );
  GSampled = sampleGConv(k,X,Y,indVol, fastconv)[:,indVol];

  push!(Entries,GSampled);

  # for  x = xmin, y = 0
  indFz1 = round(Integer, n*(m-1)/2 +1 + [0,1,n,n+1,-n, -n+1]);
  GSampled = sampleGConv(k,X,Y,indFz1, fastconv)[:,indFz1];

  push!(Entries,GSampled);

  # for  x = xmax, y = 0
  indFz2 = round(Integer, n*(m-1)/2 + [-1,0,n,n-1,-n, -n-1]);
  GSampled = sampleGConv(k,X,Y,indFz2, fastconv)[:,indFz2];

  push!(Entries,GSampled);

  # for  y = ymin, x = 0
  indFx1 = round(Integer, (n+1)/2 + [-1,0,1,n,n+1, n-1]);
  GSampled = sampleGConv(k,X,Y,indFx1, fastconv)[:,indFx1];

  push!(Entries,GSampled);


  # for  y = ymin, x = 0
  indFx2 = round(Integer, N - (n+1)/2 + [-1,0,1,-n,-n+1, -n-1]);
  GSampled = sampleGConv(k,X,Y,indFx2, fastconv)[:,indFx2];

  push!(Entries,GSampled);

  # For the corners
  indcorner1 = round(Integer, 1 + [0,1, n,n+1]);
  indcorner2 = round(Integer, n + [0,-1, n,n-1]);
  indcorner3 = round(Integer, n*m-n+1 + [0,1, -n,-n+1]);
  indcorner4 = round(Integer, n*m + [0,-1, -n,-n-1]);

  GSampled = sampleGConv(k,X,Y,indcorner1, fastconv)[:,indcorner1];
  push!(Entries,GSampled);


  #'
  GSampled = sampleGConv(k,X,Y,indcorner2, fastconv)[:,indcorner2];
  push!(Entries,GSampled);

  #'
  GSampled = sampleGConv(k,X,Y,indcorner3, fastconv)[:,indcorner3];
  push!(Entries,GSampled);

  #'
  GSampled = sampleGConv(k,X,Y,indcorner4, fastconv)[:,indcorner4];
  push!(Entries,GSampled);

  return Entries
end


function buildSparseAG(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex{Float64}, n::Int64 ,m::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method=="normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    Entries = entriesSparseG(k,X,Y,D0, n ,m);

    ValuesAG = Values[1]*Entries[1];
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                    Indices[1][:], ValuesAG[:]);

    ValuesAG = Values[2]*Entries[2];
    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:], ValuesAG[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    ValuesAG = Values[3]*Entries[3];
    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[4]*Entries[4];
    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[5]*Entries[5];
    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[6]*Entries[6];
    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[7]*Entries[7];
    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[8]*Entries[8];
    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[9]*Entries[9];
    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    AG = sparse(rowA,colA,valA);

    return AG;
end


function buildSparseAGConv( k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                            fastconv::FastM, n::Int64 ,m::Int64;
                            method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method=="normal"
      (Indices, Values) = entriesSparseAConv(k,X,Y,fastconv, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,fastconv, n ,m);
    end

    Entries = entriesSparseGConv(k,X,Y,fastconv, n ,m);

    ValuesAG = Values[1]*Entries[1];
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1][:],
                                    Indices[1][:], ValuesAG[:]);

    ValuesAG = Values[2]*Entries[2];
    (Row, Col, Val) = createIndices(Ind[1,2:end-1][:],
                                    Indices[2][:], ValuesAG[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    ValuesAG = Values[3]*Entries[3];
    (Row, Col, Val) = createIndices(Ind[end,2:end-1][:],
                                    Indices[3][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[4]*Entries[4];
    (Row, Col, Val) = createIndices(Ind[2:end-1,1][:],
                                    Indices[4][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[5]*Entries[5];
    (Row, Col, Val) = createIndices(Ind[2:end-1,end][:],
                                    Indices[5][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[6]*Entries[6];
    (Row, Col, Val) = createIndices(Ind[1,1],
                                    Indices[6][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[7]*Entries[7];
    (Row, Col, Val) = createIndices(Ind[end,1],
                                    Indices[7][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[8]*Entries[8];
    (Row, Col, Val) = createIndices(Ind[1,end],
                                    Indices[8][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    ValuesAG = Values[9]*Entries[9];
    (Row, Col, Val) = createIndices(Ind[end,end],
                                    Indices[9][:], ValuesAG[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    AG = sparse(rowA,colA,valA);

    return AG;
end



function entriesSparseA3D(k,X,Y,Z,D0, n ,m, l)
  # in this case we need to build everythig with ranodmized methods
  # we need to have an odd number of points
  #@assert mod(length(X),2) == 1
  Entries  = Array{Complex{Float64}}[]
  Indices  = Array{Int64}[]

  N = n*m*l;

  Ind_relative = zeros(Int64,3,3,3)
  Ind_relative[:,:,1] = [(-m*n-n-1) (-m*n-n) (-m*n-n+1);
                         (-m*n  -1) (-m*n  ) (-m*n  +1);
                         (-m*n+n-1) (-m*n+n) (-m*n+n+1) ]';

  Ind_relative[:,:,2] = [-n-1 -n -n+1;
                           -1  0    1;
                          n-1  n  n+1]';

  Ind_relative[:,:,3] = [(m*n-n-1) (m*n-n) (m*n-n+1);
                         (m*n  -1) (m*n  ) (m*n  +1);
                         (m*n+n-1) (m*n+n) (m*n+n+1) ]';

  # computing the entries for the interior

  nHalf = round(Integer,n/2);
  mHalf = round(Integer,m/2);
  lHalf = round(Integer,l/2);

  indVol = round(Integer, changeInd3D(nHalf,mHalf,lHalf,n,m,l)+Ind_relative[:]);

  indVolC = setdiff(collect(1:N),indVol);
  GSampled = sampleG3D(k,X,Y,Z,indVol, D0)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG3D(k,X,Y,Z,indFx1, D0)[:,indC ];
   # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,:,:][:] );#'


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  indC = setdiff(collect(1:N),indFxN);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices, Ind_relative[1:2,:,:][:]); #'


  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  indC = setdiff(collect(1:N),indFy1);
  GSampled = sampleG3D(k,X,Y,Z,indFy1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,2:3,:][:]); #'

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  indC = setdiff(collect(1:N),indFyN);
  GSampled = sampleG3D(k,X,Y,Z,indFyN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,1:2,:][:] ); #'

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG3D(k,X,Y,Z,indFz1, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,2:3][:] ); #'

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  indC = setdiff(collect(1:N),indFzN);
  GSampled = sampleG3D(k,X,Y,Z,indFzN, D0)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,1:2][:] ); #'

  # we need to incorporate the vertices
  indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
  indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
  indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
  indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
  indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
  indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
  indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
  indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
  indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
  indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
  indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
  indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));


  indC = setdiff(collect(1:N),indvertex1);
  GSampled = sampleG3D(k,X,Y,Z,indvertex1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,3,2,Ind_relative)); #'


  indC = setdiff(collect(1:N),indvertex2);
  GSampled = sampleG3D(k,X,Y,Z,indvertex2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,3,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex3);
  GSampled = sampleG3D(k,X,Y,Z,indvertex3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex4);
  GSampled = sampleG3D(k,X,Y,Z,indvertex4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex5);
  GSampled = sampleG3D(k,X,Y,Z,indvertex5, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex6);
  GSampled = sampleG3D(k,X,Y,Z,indvertex6, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex7);
  GSampled = sampleG3D(k,X,Y,Z,indvertex7, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex8);
  GSampled = sampleG3D(k,X,Y,Z,indvertex8, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex9);
  GSampled = sampleG3D(k,X,Y,Z,indvertex9, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex10);
  GSampled = sampleG3D(k,X,Y,Z,indvertex10, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex11);
  GSampled = sampleG3D(k,X,Y,Z,indvertex11, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex12);
  GSampled = sampleG3D(k,X,Y,Z,indvertex12, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,1,Ind_relative)); #'


  # Now we incorporate the corners
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  indC = setdiff(collect(1:N),indcorner1);
  GSampled = sampleG3D(k,X,Y,Z,indcorner1, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG3D(k,X,Y,Z,indcorner2, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG3D(k,X,Y,Z,indcorner3, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG3D(k,X,Y,Z,indcorner4, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,2:3][:]); #'

  indC = setdiff(collect(1:N),indcorner5);
  GSampled = sampleG3D(k,X,Y,Z,indcorner5, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner6);
  GSampled = sampleG3D(k,X,Y,Z,indcorner6, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner7);
  GSampled = sampleG3D(k,X,Y,Z,indcorner7, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner8);
  GSampled = sampleG3D(k,X,Y,Z,indcorner8, D0)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,1:2][:]); #'

  return (Indices, Entries)
end




function buildSparseA(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                       D0::Complex{Float64}, n::Int64 ,m::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method == "normal"
      (Indices, Values) = entriesSparseA(k,X,Y,D0, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end


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



function buildSparseAConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          fastconv::FastM, n::Int64 ,m::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m),n,m);

    if method == "normal"
      (Indices, Values) = entriesSparseAConv(k,X,Y,fastconv, n ,m);
    elseif method == "randomized"
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end


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
