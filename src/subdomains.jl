# file to define a subdomain for solving the Lippmann Schwinger equation
type Subdomain
    n::Int64
    m::Int64
    h::Float64
    delta::Float64 # delta for the smooth cut-off
    # indices for extracting the local information
    ind_0::Array{Int64,1}
    ind_1::Array{Int64,1}
    ind_n::Array{Int64,1}
    ind_np::Array{Int64,1}
    Hinv
    H::SparseMatrixCSC{Complex128,Int64}
    x
    y
    ylim
    indVolInt
    indVol
    indVolIntLocal
    function Subdomain(As,AG,Msp, x,y, ind1, indn, ndelta, h, nu,k)
        println("Building a subdomain")
        indStart   = ind1==1? 1 : ind1-ndelta;
        indFinish = indn==length(y) ? length(y) : ind1+ndelta;
        y1 = y[indStart : indFinish]
        x1 = copy(x)
        n1 = length(x1)
        m1 = length(y1)
        N1 = n1*m1
        X1 = repmat(x1, 1, m1)[:]
        Y1 = repmat(y1', n1,1)[:]
        X = repmat(x, 1, m)[:]
        Y = repmat(y', n,1)[:]
        ylim = [y[ind1] y[indn]]
        #the domain is decomposed in two subdomains
        println("Defining the filter")
        fcut(y) = e.^(-1./y).*(y.>0)
        fcut(y,alpha) = fcut(y)./(fcut(y) + fcut(alpha - y))
        midy = (ylim[2]+ ylim[1])/2
        widthy = abs(ylim[2] - ylim[1])/2
        width = h*(ndelta-1);
        filter(x,y) =  0*x + (y.<=(ylim[2]+h)).*(y.>=(ylim[1]-h)).*(1- fcut(abs(y - width) - witdhy-h, width))  ; #add the rest of the filter in here
        nu1(x,y) = filter1(x,y).*nu(x,y);

        println("Getting the Indices")
        indVolInt =  find((Y.>=y[ind1]).*(Y.<= y[indn]))
        indVol    =  find((Y.>=y1[1]).*(Y.<= y1[end]))
        indVolIntLocal =  find(Y1.< h)

        println("Building the local matrices")
        Mapproxsp =  (As + k^2*(AG*spdiagm(nu1(X,Y))))[indVol,indVol];

        println("Adding the Boundary conditions")
        if ind1 != 1
            println("Adding the Boundary conditions at the top")
            Mapproxsp[1:2*n, 1:2*n] = Msp[1:2*n, 1:2*n];
        elseif indn != length(y)
            println("Adding the Boundary conditions at the bottom")
            Mapproxsp[end-2*n1+1:end, end-2*n1+1:end]=Msp[end-2*n1+1:end, end-2*n1+1:end];
        end

        println("Obtaining the indices for the traces")
        if ind1 != 1
            ind_0  = find(Y1.== y[ndelta]);
            ind_1  = find(Y1.== y[ndelta+1]);
        else
            ind_0  = [];
            ind_1  = [];
        end
        if indn != length(y)
            ind_n  = find(Y1.== y[m1-ndelta]);
            ind_np = find(Y1.== y[m1-ndelta+1]);
        else
            ind_n  = [];
            ind_np  = [];
        end

        println("Factorizing the local matrix")
        Minv = lufact(Mapproxsp);

        new(n1,m1, h, ndelta*h,ind_0, ind_1, ind_n, ind_np, Minv,Mapproxsp ,
            x, y, [y[ind1] y[indn]], indVolInt, indVol,
            indVolIntLocal)
    end
end

function solve(subdomain::Subdomain, f)
    # u = solve(subdomain::Subdomain, f)
    # function that solves the system Hu=f in the subdomain
    # check size
    if (size(f[:])[1] == subdomain.n*subdomain.m)
        u = subdomain.Hinv\f[:];
        return u
    else
        print("The dimensions do not match \n");
        return 0
    end
end

function extractBoundaryData(subdomain::Subdomain, u)
    # Function to extract the boundary data from a solution u
    # input   subdomain: subdomain associated to the solution
    #         u        : solution
    # output  (u0, u1, uN, uNp) : tuple of the solution at different depth
    # check size
    if (size(u[:])[1] == subdomain.n*subdomain.m)
        u0  = u[subdomain.ind_0];
        u1  = u[subdomain.ind_1];
        uN  = u[subdomain.ind_n];
        uNp = u[subdomain.ind_np];
        return (u0,u1,uN,uNp)
    else
        print("Dimension mismatch \n");
        return 0
    end
end

function applyBlockOperator(subdomain::Subdomain,v0,v1,vN,vNp)
    # function to apply the local matricial operator to the interface data
    # and we sample it at the interface
    # allocating the source
    f = zeros(Complex{Float64},subdomain.n,subdomain.m);
    # filling the source with the correct single and double layer potentials
    # TO DO: put the correct operators in here (we need to multiply the traces)
    # by the correct weights
    f[subdomain.ind_0 ] =  v0;
    f[subdomain.ind_1 ] = -v1;
    f[subdomain.ind_n ] = -vN;
    f[subdomain.ind_np] =  vNp;
    f = f*(1/subdomain.h)^2;
    u = solve(subdomain, f[:]);

    u0  = u[subdomain.ind_0  ];
    u1  = u[subdomain.ind_1  ];
    uN  = u[subdomain.ind_n  ];
    uNp = u[subdomain.ind_np ];
    return (u0,u1,uN,uNp)

end
