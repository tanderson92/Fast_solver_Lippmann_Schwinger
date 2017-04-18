# File containing all the preconditioners


type doubleGSPreconditioner
    n::Int64  # number of grid points in the x direction
    nSubs::Int64
    subDomains1
    subDomains2
    Msp::SparseMatrixCSC{Complex{Float64},Int64}
    T::SparseMatrixCSC{Complex{Float64},Int64} # tranposition matrix
    Tt::SparseMatrixCSC{Complex{Float64},Int64} # the transpose of T
    tol::Float64
    precondtype # inner GMRES loop or not
    function doubleGSPreconditioner(subDomains1, subDomains2, Msp; tol = 1e-4, precondtype ="GS")
        n = subDomains1[1].n
        N = n*n; # we suppose that the domain is squared
        T = speye(N);
        index = 1:N;
        index = (reshape(index, n,n).')[:];
        T = T[index,:];
        nSubs = length(subDomains1)
        new(n,nSubs, subDomains1,subDomains2,Msp, T,T.', tol,precondtype) # don't know if it's the best answer
    end
end


type SparsifyingPreconditioner
    Msp::SparseMatrixCSC{Complex{Float64},Int64}
    As::SparseMatrixCSC{Complex{Float64},Int64} # tranposition matrix
    MspInv
    solverType::String
    function SparsifyingPreconditioner(Msp, As; solverType::String="UMFPACK")
        tic();
        if  solverType=="UMFPACK"
            MspInv = lufact(Msp)
            println("time for the factorization was ", toc() )
        end
        if  solverType=="MKLPARDISO"
            MspInv = MKLPardisoSolver();
            set_nprocs!(MspInv, 16)
            #setting the type of the matrix
            set_matrixtype!(MspInv,3)
            # setting we are using a transpose
            set_iparm!(MspInv,12,2)
            # setting the factoriation phase
            set_phase!(MspInv, 12)
            X = zeros(Complex128, size(Msp)[2],1)
            # factorizing the matrix
            pardiso(MspInv,X, Msp,X)
            # setting phase and parameters to solve and transposing the matrix
            # this needs to be done given the different C and Fortran convention
            # used by Pardiso (C convention) and Julia (Fortran Convention)
            set_phase!(MspInv, 33)
            set_iparm!(MspInv,12,2)
        end
        new(Msp,As, MspInv, solverType) # 
    end
end



type PolarizedTracesPreconditioner
    As::SparseMatrixCSC{Complex{Float64},Int64}
    subDomains
    IntegralPreconditioner
    tol::Float64
    nIt::Int64
    precondtype
    mkl_sparseBlas
     function PolarizedTracesPreconditioner(As, subDomains; nIt = 1, tol = 1e-2, precondtype ="GS",
                                            mkl_sparseBlas=false)
        IntPrecond = IntegralPreconditioner(subDomains, nIt = nIt, precondtype = precondtype)
        new(As, subDomains, IntPrecond, tol, nIt, precondtype, mkl_sparseBlas) # don't know if it's the best answer
    end
end


type GSPreconditioner
    n::Int64  # number of grid points in the x direction
    nSubs::Int64
    subDomains
    tol::Float64
    precondtype # preconditioner type Jacobi, GaussSeidel Opt
    function GSPreconditioner(subDomains; tol = 1e-4, precondtype ="GS")
        n = subDomains[1].n
        nSubs = length(subDomains)
        new(n,nSubs, subDomains, tol,precondtype) # don't know if it's the best answer
    end
end


type doublePreconditioner
    # Type definition for the bidirectional preconditioner
    As::SparseMatrixCSC{Complex{Float64},Int64}    # Sparsifying matrix
    Msp::SparseMatrixCSC{Complex{Float64},Int64}   # Sparsified system to solve
    doubleGSPreconditioner                         # bi-directional preconditioner
    mkl_sparseBlas                                 # flag to use sparseBLAS
    maxIter::Int64                                 # maximum number of inner iterations
    tol::Float64                                   # tolerance for the inner system 
    function doublePreconditioner(As,Msp,subDomains1,subDomains2; mkl_sparseBlas=false,  maxIter = 20, tol = 1e-2)
        new(As,Msp, doubleGSPreconditioner(subDomains1, subDomains2, Msp), mkl_sparseBlas, maxIter,tol) # don't know if it's the best answer
    end

end


type Preconditioner
    As::SparseMatrixCSC{Complex{Float64},Int64}
    Msp::SparseMatrixCSC{Complex{Float64},Int64}
    GSPreconditioner
    mkl_sparseBlas
    function Preconditioner(As,Msp,subDomains; mkl_sparseBlas=false)
        new(As,Msp, GSPreconditioner(subDomains), mkl_sparseBlas) # don't know if it's the best answer
    end

end

# Encapsulation of the preconditioner in order to use preconditioned GMRES
import Base.\

function \(M::doubleGSPreconditioner, b::Array{Complex128,1})
    #println("Applying the polarized Traces Preconditioner")

    # TODO add more options to the type of preconditioner used
    #return precondGS(M.subDomains, b)
    u = precondGSOptimized(M.subDomains1, b)
    err = M.Msp*u - b;
    u2 = (M.Tt)*(precondGSOptimized(M.subDomains2, M.T*err))
    return u-u2
end

function \(M::SparsifyingPreconditioner, b::Array{Complex128,1})
    #println("Applying the polarized Traces Preconditioner")

    # TODO add more options to the type of preconditioner used
    #return precondGS(M.subDomains, b)
    if M.solverType=="UMFPACK"
        return M.MspInv\(M.As*b)
    elseif  M.solverType=="MKLPARDISO"
        set_phase!(M.MspInv, 33)
        u = zeros(Complex128,length(b))
        pardiso(M.MspInv, u, M.Msp, M.As*b)
        return u
    end
end


function \(M::GSPreconditioner, b::Array{Complex128,1})
    #println("Applying the polarized Traces Preconditioner")

    # TODO add more options to the type of preconditioner used
    #return precondGS(M.subDomains, b)
    return precondGSOptimized(M.subDomains, b)
end

function \(M::Preconditioner, b::Array{Complex128,1})
    #println("Applying the sparsifying Preconditioner")

    y = zeros(b)
    # small
    if M.mkl_sparseBlas
        x0 = zeros(b);
        beta = Complex128(1+0im)
        SparseBLAS.cscmv!('N',beta,"GXXF",M.As,b,beta,x0)
        #println("using sparse blas!")
    else
        x0 =  M.As*b;
    end
    info = gmres!(y, M.Msp, x0 , M.GSPreconditioner, restart = 20, maxiter = 1, tol = 1e-4)
    println("Number of iterations for inner problem is ", countnz(info[2].residuals[:]))

    #y = M.GSPreconditioner\(M.As*b)
    return y
end


function \(M::PolarizedTracesPreconditioner, b::Array{Complex128,1})
    #println("Applying the sparsifying Preconditioner")

    f = extractRHS(M.subDomains, M.As*b);
    fPol = -vectorizePolarizedBdyDataRHS(M.subDomains, f);
    uPol = zeros(fPol)

    info = gmres!(uPol,x->(applyMMOptUmf(M.subDomains, x)), fPol, M.IntegralPreconditioner, tol = M.tol)

    println("Number of iterations for inner problem is ", countnz(info[2].residuals[:]))

    u = uPol[1:round(Integer, end/2)] + uPol[round(Integer, end/2)+1:end]
    
    (v0,v1,vn,vnp) = devectorizeBdyDataContiguous(M.subDomains, u)

    U = reconstruction(M.subDomains, M.As*b, v0, v1, vn, vnp);

    return U[:]
end


function \(M::doublePreconditioner, b::Array{Complex128,1})
    #println("Applying the sparsifying Preconditioner")

    if M.maxIter != 0
        y = zeros(b)
        # small if block to use sparse MKL
        if M.mkl_sparseBlas
            x0 = zeros(b);
            beta = Complex128(1+0im)
            SparseBLAS.cscmv!('N',beta,"GXXF",M.As,b,beta,x0)
            #println("using sparse blas!")
        else
            x0 =  M.As*b;
        end
            # otherwise we can just use GMRES (the overhead is enough)
            info = gmres!(y, M.Msp, x0 , M.doubleGSPreconditioner, restart = M.maxIter, maxiter = 1, tol = M.tol)
            println("Number of iterations for inner problem is ", countnz(info[2].residuals[:]))
    else
        y = M.doubleGSPreconditioner\(M.As*b)
    end
    return y
end

# Gauss Seidel preconditioner
# as explained in the paper (this needs 5 solves per iteration)
function precondGS(subDomains, source::Array{Complex128,1})
    ## We are only implementing the Jacobi version of the preconditioner
    nSubs = length(subDomains);

    n = subDomains[1].n
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].n*subDomains[ii].m) for ii = 1:nSubs ]

    # copying the wave-fields
    for ii = 1:nSubs
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]
    end

    uLocalArray = [solve(subDomains[ii],rhsLocal[ii] )  for  ii = 1:nSubs]

    u_0  = zeros(Complex128,n*nSubs)
    u_1  = zeros(Complex128,n*nSubs)
    u_n  = zeros(Complex128,n*nSubs)
    u_np = zeros(Complex128,n*nSubs)

    index = 1:n

    # Downward sweep

    u_n[index]  = uLocalArray[1][subDomains[1].ind_n]
    u_np[index] = uLocalArray[1][subDomains[1].ind_np]

    # populate the traces

    for ii = 1:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        if ii != 1
            u_0[(ii-1)*n + index] = -uLocalArray[ii][ind_0]
            u_1[(ii-1)*n + index] = -uLocalArray[ii][ind_1]
        end
        if ii !=nSubs
            u_n[(ii-1)*n  + index] = -uLocalArray[ii][ind_n]
            u_np[(ii-1)*n + index] = -uLocalArray[ii][ind_np]
        end

    end


    u_n[ index] = -u_n[ index]
    u_np[index] = -u_np[index]

    for ii = 2:nSubs-1
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_0] =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1] = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]

        uDown = solve(subDomains[ii],rhsLocaltemp)

        u_n[(ii-1)*n  + index] = -u_n[(ii-1)*n  + index] + uDown[ind_n]
        u_np[(ii-1)*n + index] = -u_np[(ii-1)*n + index] + uDown[ind_np]

    end

    #Applying L

    for ii = 2:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        if ii != nSubs

        rhsLocaltemp[subDomains[ii].ind_0]  =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1]  = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]


        rhsLocaltemp[subDomains[ii].ind_n]  = -subDomains[ii].H[ind_n,ind_np]*u_np[(ii-1)*n + index]
        rhsLocaltemp[subDomains[ii].ind_np] =  subDomains[ii].H[ind_np,ind_n]*u_n[(ii-1)*n + index]

        else
            rhsLocaltemp[subDomains[ii].ind_0]  =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_1]  = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
        end

        uL = solve(subDomains[ii],rhsLocaltemp)

        # saving the reflections
        u_0[(ii-1)*n + index] -= uL[ind_0]
        u_1[(ii-1)*n + index] -= uL[ind_1] - u_np[(ii-2)*n + index]

    end

    # Upward sweep

     u_0[(nSubs-1)*n + index] = -u_0[(nSubs-1)*n + index]
     u_1[(nSubs-1)*n + index] = -u_1[(nSubs-1)*n + index]

     for ii = nSubs-1:-1:2
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source (we may need a sparse source)
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_np] =  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
        rhsLocaltemp[subDomains[ii].ind_n]  = -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]

        uUp = solve(subDomains[ii],rhsLocaltemp)

        u_0[(ii-1)*n + index] = -u_0[(ii-1)*n + index] + uUp[ind_0]
        u_1[(ii-1)*n + index] = -u_1[(ii-1)*n + index] + uUp[ind_1]

    end

    # reconstruction

    uPrecond = Complex128[]

    for ii = 1:nSubs

        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        # adding the source at the boundaries
        if ii!= 1
            # we need to be carefull at the edges
            rhsLocaltemp[subDomains[ii].ind_1]  += -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_0]  +=  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]

        end
        if ii!= nSubs
            rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
            rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]
        end

        uLocal = solve(subDomains[ii],rhsLocaltemp)

        uPrecond = vcat(uPrecond,uLocal[subDomains[ii].indVolIntLocal])
    end
    return  uPrecond
end

# Jacobi preconditioner
function precondJacobi(subDomains, source::Array{Complex128,1})
    ## We are only implementing the Jacobi version of the preconditioner
    nSubs = length(subDomains);

    n = subDomains[1].n
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].n*subDomains[ii].m) for ii = 1:nSubs ]

    # copying the wave-fields
    for ii = 1:nSubs
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]
    end

    uLocalArray = [solve(subDomains[ii],rhsLocal[ii] )  for  ii = 1:nSubs]

    u_0  = zeros(Complex128,n*nSubs)
    u_1  = zeros(Complex128,n*nSubs)
    u_n  = zeros(Complex128,n*nSubs)
    u_np = zeros(Complex128,n*nSubs)

    index = 1:n

    # Downward sweep

    u_n[index]  = uLocalArray[1][subDomains[1].ind_n]
    u_np[index] = uLocalArray[1][subDomains[1].ind_np]

    # populate the traces

    for ii = 1:nSubs
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        if ii != 1
            u_0[(ii-1)*n + index] = -uLocalArray[ii][ind_0]
            u_1[(ii-1)*n + index] = -uLocalArray[ii][ind_1]
        end
        if ii !=nSubs
            u_n[(ii-1)*n  + index] = -uLocalArray[ii][ind_n]
            u_np[(ii-1)*n + index] = -uLocalArray[ii][ind_np]
        end

    end

    u_n[ index] = -u_n[ index]
    u_np[index] = -u_np[index]

    for ii = 2:nSubs-1
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the partitioned source
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_0] =  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]
        rhsLocaltemp[subDomains[ii].ind_1] = -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]

        uDown = solve(subDomains[ii],rhsLocaltemp)

        u_n[(ii-1)*n  + index] = -u_n[(ii-1)*n  + index] + uDown[ind_n]
        u_np[(ii-1)*n + index] = -u_np[(ii-1)*n + index] + uDown[ind_np]

    end

    # Upward sweep

     u_0[(nSubs-1)*n + index] = -u_0[(nSubs-1)*n + index]
     u_1[(nSubs-1)*n + index] = -u_1[(nSubs-1)*n + index]

     for ii = nSubs-1:-1:2
        # this will be slow most likely but it is readable
        # we will need to modify this part
        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source (we may need a sparse source)
        rhsLocaltemp = zeros(Complex128,size(rhsLocal[ii])[1]) ;

        rhsLocaltemp[subDomains[ii].ind_np] =  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
        rhsLocaltemp[subDomains[ii].ind_n]  = -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]

        uUp = solve(subDomains[ii],rhsLocaltemp)

        u_0[(ii-1)*n + index] = -u_0[(ii-1)*n + index] + uUp[ind_0]
        u_1[(ii-1)*n + index] = -u_1[(ii-1)*n + index] + uUp[ind_1]

    end

    # reconstruction

    uPrecond = Complex128[]

    for ii = 1:nSubs

        ind_0  = subDomains[ii].ind_0
        ind_1  = subDomains[ii].ind_1
        ind_n  = subDomains[ii].ind_n
        ind_np = subDomains[ii].ind_np

        # making a copy of the parititioned source
        rhsLocaltemp = copy(rhsLocal[ii]);

        # adding the source at the boundaries
        if ii!= 1
            # we need to be carefull at the edges
            rhsLocaltemp[subDomains[ii].ind_1]  += -subDomains[ii].H[ind_1,ind_0]*u_n[(ii-2)*n + index]
            rhsLocaltemp[subDomains[ii].ind_0]  +=  subDomains[ii].H[ind_0,ind_1]*u_np[(ii-2)*n + index]

        end
        if ii!= nSubs
            rhsLocaltemp[subDomains[ii].ind_np] +=  subDomains[ii].H[ind_np,ind_n]*u_0[(ii)*n + index]
            rhsLocaltemp[subDomains[ii].ind_n]  += -subDomains[ii].H[ind_n,ind_np]*u_1[(ii)*n + index]
        end

        uLocal = solve(subDomains[ii],rhsLocaltemp)

        uPrecond = vcat(uPrecond,uLocal[suDbomains[ii].indVolIntLocal])
    end
    return  uPrecond
end


# Optimized Gauss-Seidel Preconditioner
# this one uses the Polarization conditions to reduce the number of
# solves per iteration (we went from 5 to only two)
# This version reduces the allocations by 10%
function precondGSOptimized(subDomains, source::Array{Complex128,1})
    ## We are only implementing the Gauss-Seidel version of the preconditioner
    nSubs = length(subDomains);

    n = subDomains[1].n
    # building the local rhs
    rhsLocal = [ zeros(Complex128,subDomains[ii].n*subDomains[ii].m) for ii = 1:nSubs ]

    localSizes = zeros(Int64,nSubs)

    # allocating spzce for the boundary data
    u_0  = zeros(Complex128,n*nSubs)
    u_1  = zeros(Complex128,n*nSubs)
    u_n  = zeros(Complex128,n*nSubs)
    u_np = zeros(Complex128,n*nSubs)

    index = 1:n

    # Local solves + Downward sweep
    for ii = 1:nSubs

        # obtaining the local sources
        rhsLocal[ii][subDomains[ii].indVolIntLocal] = source[subDomains[ii].indVolInt]

        # we obtain the local sizes for each subdomain
        localSizes[ii] = length(subDomains[ii].indVolIntLocal)

        # adding the equivalent sources for the one-sided GRF
        if ii !=1
            rhsLocal[ii][subDomains[ii].ind_0] +=  subDomains[ii].H[subDomains[ii].ind_0,subDomains[ii].ind_1]*u_np[(ii-2)*n + index]
            rhsLocal[ii][subDomains[ii].ind_1] += -subDomains[ii].H[subDomains[ii].ind_1,subDomains[ii].ind_0]*u_n[(ii-2)*n + index]
        end

        # solving the rhs
        vDown = solve(subDomains[ii],rhsLocal[ii])

        # extracting the traces
        if ii != nSubs
            u_n[(ii-1)*n  + index] = vDown[subDomains[ii].ind_n]
            u_np[(ii-1)*n + index] = vDown[subDomains[ii].ind_np]
        end

    end

    # obtaining the limit of each subdomain within the global approximated solution
    localLim = [0; cumsum(localSizes)];

    # Upward sweep + reflections + reconstruction
    # allocating space for the solution
    uPrecond = zeros(Complex128, length(source))

    for ii = nSubs:-1:1

        # adding the equivalent sources for the one-sided GRF
        if ii!= nSubs
            rhsLocal[ii][subDomains[ii].ind_np] +=  subDomains[ii].H[subDomains[ii].ind_np,subDomains[ii].ind_n]*u_0[(ii)*n + index]
            rhsLocal[ii][subDomains[ii].ind_n]  += -subDomains[ii].H[subDomains[ii].ind_n ,subDomains[ii].ind_np]*u_1[(ii)*n + index]
        end

        # solving the local problem
        uUp = solve(subDomains[ii],rhsLocal[ii])

        # extracting the data for the next subdomains and adding the reflections
        if ii > 1
            u_0[(ii-1)*n + index] = uUp[subDomains[ii].ind_0]
            u_1[(ii-1)*n + index] = uUp[subDomains[ii].ind_1] - u_np[(ii-2)*n + index]
        end

        # reconstructing the problem on the fly
        uPrecond[localLim[ii]+1:localLim[ii+1]] = uUp[subDomains[ii].indVolIntLocal]
    end

    return  uPrecond
end

