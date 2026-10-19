function    [D]    =  Nonnegative_DL( X, par )
    rng(0);
    Q            =    randperm( size(X,2) );
    D            =    X(:, Q(1:par.K));
    D            =    D./repmat((sqrt(sum(D.^2))+eps), size(D,1), 1);

    Iter         =    10;
    fun          =    zeros(Iter, 1);
    X_s          =    X;
    par.lambda   =    0.001;

    for  t   =  1 : Iter

        B     =    Nonnegative_SC( D, X_s, par );
        b     =    sum(B.^2, 2);
        R     =    X - D*B;

        for k = 1 : par.K
            d_k_pre   =   D(:,k);
            d_k       =   max( D(:,k) + R*B(k,:)'/b(k), 0 );
            d_k       =   d_k./max(norm(d_k), 1);
            D(:,k)    =   d_k;
       
            R0        =   R;
            R         =   R0 - (d_k - d_k_pre)*B(k,:);
        end
    
        fun(t)   =   0.5*sum(sum((X_s-D*B).^2)) + par.lambda*sum( sum( abs(B) ) );
    end




function  A   =  Nonnegative_SC( D, X, par )

    A        =    zeros( size(D,2), size(X, 2) );
    V        =    zeros( size(D,2), size(X, 2) );
    T        =    100;
    fun      =    zeros(T,1);
    DTD       =   D'*D;
    DTX       =   D'*X;
    Ek        =   eye(par.K);
    mu        =   0.01;
    ro        =   1.07;
    for  i  =  1:T
        S         =   (DTD + mu*Ek)\(DTX + mu*(A-V/(2*mu)) );
        A         =   max( soft(S+V/(2*mu), par.lambda/(2*mu)), 0);
        V         =   V + 2*mu*( S - A );
        mu        =   mu*ro;

        fun(i)    =   0.5*sum(sum((X-D*A).^2)) + par.lambda*sum(sum(abs(A)));
    end

function y = soft(x,tau)

    y = sign(x).*max(abs(x)-tau,0);


function B = repmat(A, M, varargin)

    try
        B = matlab.internal.builtinhelper.repmat(A, M, varargin{:});
    catch ME
        throw(ME);
    end


