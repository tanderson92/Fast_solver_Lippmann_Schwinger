## Functions to build the sparsifying matrices in 3D 

include("FastConvolution3D.jl")


## helper functions to deal with the indeces
@inline function changeInd3D(indI::Int64, indJ::Int64, indK::Int64, n::Int64,m::Int64,k::Int64)
  # function easily compute the global index from dimensional ones
  # for a third order tensor
  return (indK-1)*n*m + (indJ-1)*n + indI
end

@inline function changeInd3D(indI::Array{Int64,1}, indJ::Array{Int64,1}, indK::Array{Int64,1}, n::Int64,m::Int64,k::Int64)
  # function easily compute the global index from dimensional ones
  # for a third order tensor
  return (indK-1)*n*m + (indJ-1)*n + indI
end

@inline function subStencil3D(indI::Int64, indJ::Int64, indK::Int64, Ind_relative::Array{Int64,3})
  # function to easily compute the local substencil for a given set of indices!
  # for a third order tensor
  ii = max(indI-1,1):min(indI+1,3)
  jj = max(indJ-1,1):min(indJ+1,3)
  kk = max(indK-1,1):min(indK+1,3)

  return Ind_relative[ii,jj,kk][:]
end



## To be done! (I don't remember how I built this one)
function buildSparseA3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                        Z::Array{Float64,1},
                        D0::Complex128, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner


    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,D0, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], Values[10][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], Values[11][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], Values[12][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], Values[13][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], Values[14][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], Values[15][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], Values[16][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], Values[17][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], Values[18][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], Values[19][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], Values[20][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], Values[21][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], Values[22][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], Values[23][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], Values[24][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], Values[25][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], Values[26][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], Values[27][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end



function entriesSparseG3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          D0::Complex128, n::Int64 ,m::Int64,l::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  #

  Entries  = Array{Complex128}[]

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

  GSampled = sampleG3D(k,X,Y,Z,indVol, D0)[:,indVol];

  push!(Entries,GSampled);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFx1, D0)[:,indFx1];

  push!(Entries,GSampled);


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, D0)[:,indFxN];

  push!(Entries,GSampled);

  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFy1, D0)[:,indFy1];

  push!(Entries,GSampled);

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFyN, D0)[:,indFyN];

  push!(Entries,GSampled);

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFz1, D0)[:,indFz1];

  push!(Entries,GSampled);

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFzN, D0)[:,indFzN];

  push!(Entries,GSampled);

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

  GSampled = sampleG3D(k,X,Y,Z,indvertex1, D0)[:,indvertex1 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex2, D0)[:,indvertex2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex3, D0)[:,indvertex3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex4, D0)[:,indvertex4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex5, D0)[:,indvertex5 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex6, D0)[:,indvertex6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex7, D0)[:,indvertex7 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex8, D0)[:,indvertex8 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex9, D0)[:,indvertex9 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex10, D0)[:,indvertex10 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex11, D0)[:,indvertex11 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex12, D0)[:,indvertex12 ];
  push!(Entries,GSampled);


  # Now we incorporate the corners
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  GSampled = sampleG3D(k,X,Y,Z,indcorner1, D0)[:,indcorner1 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner2, D0)[:,indcorner2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner3, D0)[:,indcorner3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner4, D0)[:,indcorner4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner5, D0)[:,indcorner5 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indcorner6, D0)[:,indcorner6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner7, D0)[:,indcorner7 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner8, D0)[:,indcorner8 ];
  push!(Entries,GSampled);

  return Entries

end

function buildSparseAG3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          D0::Complex128, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsifying preconditioner

    Entries = entriesSparseG3D(k,X,Y,Z,D0, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,D0, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    ValuesAG = Values[1]*Entries[1];


    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)





# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)



    AG = sparse(rowA,colA,valA);

    return AG;
end




function buildSparseAG3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    ValuesAG = Values[1]*Entries[1];


    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)





# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)



    AG = sparse(rowA,colA,valA);

    return AG;
end



function entriesSparseG3D(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64)
  # function to compute the entried of G, inside the volume, at the boundaries
  # and at the corners. This allows us to compute A*G in O(n) time instead of
  # O(n^2)
  # we need to have an even number of points
  #

  Entries  = Array{Complex128}[]

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

  GSampled = sampleG3D(k,X,Y,Z,indVol, fastconv)[:,indVol];

  push!(Entries,GSampled);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFx1, fastconv)[:,indFx1];

  push!(Entries,GSampled);


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, fastconv)[:,indFxN];

  push!(Entries,GSampled);

  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFy1, fastconv)[:,indFy1];

  push!(Entries,GSampled);

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFyN, fastconv)[:,indFyN];

  push!(Entries,GSampled);

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFz1, fastconv)[:,indFz1];

  push!(Entries,GSampled);

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  GSampled = sampleG3D(k,X,Y,Z,indFzN, fastconv)[:,indFzN];

  push!(Entries,GSampled);

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

  GSampled = sampleG3D(k,X,Y,Z,indvertex1, fastconv)[:,indvertex1 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex2, fastconv)[:,indvertex2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex3, fastconv)[:,indvertex3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex4, fastconv)[:,indvertex4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex5, fastconv)[:,indvertex5 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex6, fastconv)[:,indvertex6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex7, fastconv)[:,indvertex7 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex8, fastconv)[:,indvertex8 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex9, fastconv)[:,indvertex9 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indvertex10, fastconv)[:,indvertex10 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex11, fastconv)[:,indvertex11 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indvertex12, fastconv)[:,indvertex12 ];
  push!(Entries,GSampled);


  # Now we incorporate the corners
  indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
  indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
  indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
  indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
  indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
  indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
  indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
  indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);

  GSampled = sampleG3D(k,X,Y,Z,indcorner1, fastconv)[:,indcorner1 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner2, fastconv)[:,indcorner2 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner3, fastconv)[:,indcorner3 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner4, fastconv)[:,indcorner4 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner5, fastconv)[:,indcorner5 ];
  push!(Entries,GSampled);


  GSampled = sampleG3D(k,X,Y,Z,indcorner6, fastconv)[:,indcorner6 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner7, fastconv)[:,indcorner7 ];
  push!(Entries,GSampled);

  GSampled = sampleG3D(k,X,Y,Z,indcorner8, fastconv)[:,indcorner8 ];
  push!(Entries,GSampled);

  return Entries

end




function entriesSparseA3D(k,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},fastconv::FastM3D,
                          n::Int64 ,m::Int64, l::Int64)
  # in this case we need to build everythig with ranodmized methods
  # we need to have an odd number of points
  #@assert mod(length(X),2) == 1
  Entries  = Array{Complex128}[]
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
  GSampled = sampleG3D(k,X,Y,Z,indVol, fastconv)[:,indVolC ];

  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative);


  # for  x = xmin,  y = anything z = anything
  indFx1 = round(Integer, changeInd3D(1,mHalf,lHalf,n,m,l) + Ind_relative[2:3,:,:][:] );
  indC = setdiff(collect(1:N),indFx1);
  GSampled = sampleG3D(k,X,Y,Z,indFx1, fastconv)[:,indC ];
   # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,:,:][:] );#'


  # for  x = xmax, y = any z = any
  indFxN = round(Integer, changeInd3D(n,mHalf,lHalf,n,m,l) + Ind_relative[1:2,:,:][:]);
  indC = setdiff(collect(1:N),indFxN);
  GSampled = sampleG3D(k,X,Y,Z,indFxN, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices, Ind_relative[1:2,:,:][:]); #'


  # for  y = ymin, x = any z = any
  indFy1 = round(Integer, changeInd3D(nHalf,1,lHalf,n,m,l) + Ind_relative[:,2:3,:][:] );
  indC = setdiff(collect(1:N),indFy1);
  GSampled = sampleG3D(k,X,Y,Z,indFy1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,2:3,:][:]); #'

  # for  y = ymax, x = any z = any
  indFyN = round(Integer, changeInd3D(nHalf,m,lHalf,n,m,l) + Ind_relative[:,1:2,:][:] );
  indC = setdiff(collect(1:N),indFyN);
  GSampled = sampleG3D(k,X,Y,Z,indFyN, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  # saving the entries
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,1:2,:][:] ); #'

  # for  z = zmin, x = any y = any
  indFz1 = round(Integer, changeInd3D(nHalf,mHalf,1,n,m,l) + Ind_relative[:,:,2:3][:] );
  indC = setdiff(collect(1:N),indFz1);
  GSampled = sampleG3D(k,X,Y,Z,indFz1, fastconv)[:,indC ];
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[:,:,2:3][:] ); #'

  # for  z = zmax, x = any y = any
  indFzN = round(Integer, changeInd3D(nHalf,mHalf,l,n,m,l) + Ind_relative[:,:,1:2][:] );
  indC = setdiff(collect(1:N),indFzN);
  GSampled = sampleG3D(k,X,Y,Z,indFzN, fastconv)[:,indC ];
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
  GSampled = sampleG3D(k,X,Y,Z,indvertex1, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,3,2,Ind_relative)); #'


  indC = setdiff(collect(1:N),indvertex2);
  GSampled = sampleG3D(k,X,Y,Z,indvertex2, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,3,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex3);
  GSampled = sampleG3D(k,X,Y,Z,indvertex3, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex4);
  GSampled = sampleG3D(k,X,Y,Z,indvertex4, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,1,2,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex5);
  GSampled = sampleG3D(k,X,Y,Z,indvertex5, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex6);
  GSampled = sampleG3D(k,X,Y,Z,indvertex6, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex7);
  GSampled = sampleG3D(k,X,Y,Z,indvertex7, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(3,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex8);
  GSampled = sampleG3D(k,X,Y,Z,indvertex8, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(1,2,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex9);
  GSampled = sampleG3D(k,X,Y,Z,indvertex9, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex10);
  GSampled = sampleG3D(k,X,Y,Z,indvertex10, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,1,3,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex11);
  GSampled = sampleG3D(k,X,Y,Z,indvertex11, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,subStencil3D(2,3,1,Ind_relative)); #'

  indC = setdiff(collect(1:N),indvertex12);
  GSampled = sampleG3D(k,X,Y,Z,indvertex12, fastconv)[:,indC ];
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
  GSampled = sampleG3D(k,X,Y,Z,indcorner1, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner2);
  GSampled = sampleG3D(k,X,Y,Z,indcorner2, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner3);
  GSampled = sampleG3D(k,X,Y,Z,indcorner3, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,2:3][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner4);
  GSampled = sampleG3D(k,X,Y,Z,indcorner4, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,2:3][:]); #'

  indC = setdiff(collect(1:N),indcorner5);
  GSampled = sampleG3D(k,X,Y,Z,indcorner5, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner6);
  GSampled = sampleG3D(k,X,Y,Z,indcorner6, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,2:3,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner7);
  GSampled = sampleG3D(k,X,Y,Z,indcorner7, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[2:3,1:2,1:2][:]); #'

  #'
  indC = setdiff(collect(1:N),indcorner8);
  GSampled = sampleG3D(k,X,Y,Z,indcorner8, fastconv)[:,indC ];
  # computing the sparsifying correction
  (U,s,V) = svd(GSampled);
  push!(Entries,U[:,end]'); #'
  push!(Indices,Ind_relative[1:2,1:2,1:2][:]); #'

  return (Indices, Entries)
end


## To be done! (I don't remember how I built this one)
function buildSparseA3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                        Z::Array{Float64,1},
                        fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner
# this varians uses the kernel from the discretization of the integral system


    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,fastconv, n ,m);
    end

    # building the indices, columns and rows for the interior
    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], Values[10][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], Values[11][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], Values[12][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], Values[13][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], Values[14][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], Values[15][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], Values[16][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], Values[17][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], Values[18][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], Values[19][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], Values[20][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], Values[21][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], Values[22][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], Values[23][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], Values[24][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], Values[25][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], Values[26][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], Values[27][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)


    A = sparse(rowA,colA,valA);

    return A;
end




function buildSparseAG3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end

    ValuesAG = Values[1]*Entries[1];


    # building the indices, columns and rows for the interior
    (rowAG, colAG, valAG) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);


    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)





# we need to incorporate the vertices

# indvertex1  = round(Integer, changeInd3D(1,1,lHalf,n,m,l) + subStencil3D(3,3,2,Ind_relative));
# for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex2  = round(Integer, changeInd3D(n,1,lHalf,n,m,l) + subStencil3D(1,3,2,Ind_relative));
# for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex3  = round(Integer, changeInd3D(1,m,lHalf,n,m,l) + subStencil3D(3,1,2,Ind_relative));
# for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex4  = round(Integer, changeInd3D(n,m,lHalf,n,m,l) + subStencil3D(1,1,2,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex5  = round(Integer, changeInd3D(1,mHalf,1,n,m,l) + subStencil3D(3,2,3,Ind_relative));
# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex6  = round(Integer, changeInd3D(n,mHalf,1,n,m,l) + subStencil3D(1,2,3,Ind_relative));
# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex7  = round(Integer, changeInd3D(1,mHalf,l,n,m,l) + subStencil3D(3,2,1,Ind_relative));
# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex8  = round(Integer, changeInd3D(n,mHalf,l,n,m,l) + subStencil3D(1,2,1,Ind_relative));
# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex9  = round(Integer, changeInd3D(nHalf,1,1,n,m,l) + subStencil3D(2,3,3,Ind_relative));
# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)



    AG = sparse(rowAG,colAG,valAG);

    return AG;
end





function buildSparseAG_A3DConv(k::Float64,X::Array{Float64,1},Y::Array{Float64,1},
                          Z::Array{Float64,1},
                          fastconv::FastM3D, n::Int64 ,m::Int64,l::Int64; method::String = "normal")
# function that build the sparsigying preconditioner

    # Quick fix: TODO change this!


    Entries = entriesSparseG3D(k,X,Y,Z,fastconv, n ,m, l);
    Ind = reshape(collect(1:n*m*l),n,m,l);

    if method == "normal"
      (Indices, Values) = entriesSparseA3D(k,X,Y,Z,fastconv, n ,m,l);
    elseif method == "randomized"
      println("Not implemented yet")
      (Indices, Values) = entriesSparseARand(k,X,Y,D0, n ,m);
    end


    (rowA, colA, valA) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], Values[1][:]);

    # building the indices, columns and rows for the interior
    (rowAG, colAG, valAG) = createIndices(Ind[2:end-1,2:end-1,2:end-1][:],
                                    Indices[1][:], (Values[1]*Entries[1])[ :]);

     # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], Values[2][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,2:end-1][:],
                                    Indices[2][:], (Values[2]*Entries[2])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

     # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], Values[3][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  x = xmax,  y = anything z = anything
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,2:end-1][:],
                                    Indices[3][:], (Values[3]*Entries[3])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], Values[4][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymin,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,2:end-1][:],
                                    Indices[4][:], (Values[4]*Entries[4])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], Values[5][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  y = ymax,  x = anything z = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,2:end-1][:],
                                    Indices[5][:], (Values[5]*Entries[5])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], Values[6][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmin,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,1][:],
                                    Indices[6][:], (Values[6]*Entries[6])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], Values[7][:]);

    rowA = vcat(rowA,Row)
    colA = vcat(colA,Col)
    valA = vcat(valA,Val)

    # for  z = zmax,  x = anything y = anything
    (Row, Col, Val) = createIndices(Ind[2:end-1,2:end-1,end][:],
                                    Indices[7][:], (Values[7]*Entries[7])[ :]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)





# we need to incorporate the vertices

    # for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], Values[8][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[1,1,2:end-1][:],
                                    Indices[8][:], (Values[8]*Entries[8])[ :]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], Values[9][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymin z = anything
    (Row, Col, Val) = createIndices(Ind[end,1,2:end-1][:],
                                    Indices[9][:], (Values[9]*Entries[9])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], Values[10][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmin,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,end,2:end-1][:],
                                    Indices[10][:], (Values[10]*Entries[10])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], Values[11][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[end,end,2:end-1][:],
                                    Indices[11][:], (Values[11]*Entries[11])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], Values[12][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = ymax z = anything
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,1][:],
                                    Indices[12][:], (Values[12]*Entries[12])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], Values[13][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);
    # for  x = xmax,  y = any z = zmin
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,1][:],
                                    Indices[13][:], (Values[13]*Entries[13])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], Values[14][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);
    # for  x = xmin,  y = any z = zmax
    (Row, Col, Val) = createIndices(Ind[1,2:end-1,end][:],
                                    Indices[14][:], (Values[14]*Entries[14])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], Values[15][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = xmax,  y = any  z = zmax
    (Row, Col, Val) = createIndices(Ind[end,2:end-1,end][:],
                                    Indices[15][:], (Values[15]*Entries[15])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], Values[16][:]);

    rowA = vcat(rowA,Row);
    colA = vcat(colA,Col);
    valA = vcat(valA,Val);

    # for  x = any,  y = tmin  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,1][:],
                                    Indices[16][:], (Values[16]*Entries[16])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex10 = round(Integer, changeInd3D(nHalf,m,1,n,m,l) + subStencil3D(2,1,3,Ind_relative));
# for  x = any,  y = ymax  z = zmin
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,1][:],
                                    Indices[17][:], (Values[17]*Entries[17])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex11 = round(Integer, changeInd3D(nHalf,1,l,n,m,l) + subStencil3D(2,3,1,Ind_relative));
# for  x = any,  y = ymin  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,1,end][:],
                                    Indices[18][:], (Values[18]*Entries[18])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

# indvertex12 = round(Integer, changeInd3D(nHalf,m,l,n,m,l) + subStencil3D(2,1,1,Ind_relative));
# for  x = any,  y = ymax  z = zmax
    (Row, Col, Val) = createIndices(Ind[2:end-1,end,end][:],
                                    Indices[19][:], (Values[19]*Entries[19])[:]);

    rowAG = vcat(rowAG,Row);
    colAG = vcat(colAG,Col);
    valAG = vcat(valAG,Val);

#################################################################################
############## Now we incorporate the corners  ##################################
#################################################################################

# indcorner1 = round(Integer, changeInd3D(1,1,1,n,m,l) + Ind_relative[2:3,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,1],
                                    Indices[20][:], (Values[20]*Entries[20])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner2 = round(Integer, changeInd3D(n,1,1,n,m,l) + Ind_relative[1:2,2:3,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,1],
                                    Indices[21][:], (Values[21]*Entries[21])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner3 = round(Integer, changeInd3D(1,m,1,n,m,l) + Ind_relative[2:3,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,1],
                                    Indices[22][:], (Values[22]*Entries[22])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner4 = round(Integer, changeInd3D(n,m,1,n,m,l) + Ind_relative[1:2,1:2,2:3][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,1],
                                    Indices[23][:], (Values[23]*Entries[23])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner5 = round(Integer, changeInd3D(1,1,l,n,m,l) + Ind_relative[2:3,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,1,end],
                                    Indices[24][:], (Values[24]*Entries[24])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner6 = round(Integer, changeInd3D(n,1,l,n,m,l) + Ind_relative[1:2,2:3,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,1,end],
                                    Indices[25][:], (Values[25]*Entries[25])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner7 = round(Integer, changeInd3D(1,m,l,n,m,l) + Ind_relative[2:3,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[1,end,end],
                                    Indices[26][:], (Values[26]*Entries[26])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)
# indcorner8 = round(Integer, changeInd3D(n,m,l,n,m,l) + Ind_relative[1:2,1:2,1:2][:]);
    (Row, Col, Val) = createIndices(Ind[end,end,end],
                                    Indices[27][:], (Values[27]*Entries[27])[:]);

    rowAG = vcat(rowAG,Row)
    colAG = vcat(colAG,Col)
    valAG = vcat(valAG,Val)



    AG = sparse(rowAG,colAG,valAG);

    return AG;
end
