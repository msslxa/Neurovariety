////////////////////////////////////////////////////////////////////////////////
// NeuroVarieties.m – symbolic neuro-variety toolkit (Magma)
//
// Overview
//   A compact library to build symbolic feed-forward networks, extract the
//   coefficient map, impose gauges, and study the dimension of the resulting
//   “neuro-varieties” via Jacobian-rank sampling. Includes a small search
//   harness and a composite-Veronese + projection constructor.
//
// Main entry points
//   • BuildNetworkDataNamed(n,d)
//   • ParametrizationNamed(n,d)
//   • AffineGaugeParametrizationNamed(n,d)
//   • GaugeJacobianRankRandom(n,d)
//   • NeuroVarietyStats(n,d : tries := 10)
//   • ExhaustiveSmallDefectTest  (top-level blocks at the end of the file)
//   • Composite-Veronese and Projection Construction (final section)
//
// Conventions
//   • n = [n0,…,nL]      layer widths (inputs n0, outputs nL)
//   • d = [d1,…,d_{L-1}] coordinate-wise power activations x ↦ x^{di}
//   • K                  base field (defaults to Q via RationalField())
//   • Gauge (affine)     last column of each W_i is set to 1
//
// Quick start
//   SetColumns(1000);
//   n := [2,2,3,1];  d := [3,3];
//   phi, _ := ParametrizationNamed(n,d);
//   phiAff, _ := AffineGaugeParametrizationNamed(n,d);
//   rank, P := GaugeJacobianRankRandom(n,d);
//   stats := NeuroVarietyStats(n,d : tries := 20);
//
// Notes
//   • This file is self-contained; no external packages required.
//   • All maps return readable names for domain/codomain coordinates.
//   • Jacobian evaluation samples random integer points and avoids poles.
//
////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
//  BuildNetworkDataNamed
//  --------------------------------------------------------------
//  PURPOSE
//      Construct the *symbolic* forward pass of a feed‑forward network
//      whose layer widths are  n = [n0 … nL]  and whose activation
//      exponents are  d = [d1 … d_{L‑1}].
//
//  RETURNS
//      R   – polynomial ring  K[w…, x0,…,x_{n0-1}]  with readable names
//            (w<i>r<row>c<col> for weights,  x* for inputs).
//      Ws  – list of weight matrices W₁,…,W_L whose entries are the
//            corresponding variables in R.
//      v   – vector (F₁,…,F_{nL}) of the network‑output polynomials
//            obtained by propagating the input vector through
//            W₁ ▸ σ₁ ▸ … ▸ W_L.
//
//  STEP SUMMARY
//      1) Generate variable names and create the ring R.
//      2) Assemble each weight matrix W_i from those variables.
//      3) Perform a formal forward pass, alternating matrix products
//         and coordinate‑wise powers  ( σ_i: x ↦ x^{d_i} ).
//
//  The resulting triple (R, Ws, v) feeds all subsequent routines
//  that need explicit polynomials: coefficient extraction,
//  parametrization, Jacobian computation, &c.
//
//////////////////////////////////////////////////////////////////////
function BuildNetworkDataNamed(n , d : K := RationalField())
K := RationalField();
    if #d ne (#n - 2) then
        error "#d must equal (#n‑2)";
    end if;

    L := #n - 1;

    // --- build list of variable names ---------------------------------
    names := [* *];
    for i in [1..L] do
        for r in [0..n[i+1]-1], s in [0..n[i]-1] do
            Append(~names , Sprintf("w%or%oc%o", i,r,s));
        end for;
    end for;
    for j in [0..n[1]-1] do
        Append(~names , Sprintf("x%o", j));
    end for;
    names := [ Sprint(u) : u in names ];          // SeqEnum[String]

    // --- renamed polynomial ring --------------------------------------
    R := PolynomialRing(K , #names);
    AssignNames(~R , names);

    // --- index of wᵢ_{r,s} in R ---------------------------------------
    function Idx(i,r,s)
        ofs := 0;  for t in [1..i-1] do  ofs +:= n[t+1]*n[t];  end for;
        return ofs + r*n[i] + s + 1;
    end function;

    // --- weight matrices W_i ------------------------------------------
    Ws := [* *];
    for i in [1..L] do
        rows := n[i+1];  cols := n[i];  entries := [];
        for r in [0..rows-1], s in [0..cols-1] do
            Append(~entries , R.(Idx(i,r,s)));
        end for;
        Append(~Ws , Matrix(R, rows, cols, entries));
    end for;

    // --- input vector --------------------------------------------------
    xvars := [ R.(#names - n[1] + j + 1) : j in [0..n[1]-1] ];
    v := Vector(R , xvars);

    // --- forward pass --------------------------------------------------
    for i in [1..L] do
        v := Vector(R ,
                Eltseq(Ws[i] * Matrix(R, #Eltseq(v),1,Eltseq(v))));
        if i lt L then
            v := Vector(R , [ v[j]^d[i] : j in [1..#Eltseq(v)] ]);
        end if;
    end for;

    return R , Ws , v;
end function;

//////////////////////////////////////////////////////////////////////
//  MonomialsDegreeR
//  --------------------------------------------------------------
//  Return exponent vectors and explicit monomials of total degree g
//  in the variables  xvarsR.
//
//////////////////////////////////////////////////////////////////////
function MonomialsDegreeR(xvarsR , g)
    n0 := #xvarsR;
    Rx<x> := PolynomialRing(BaseRing(Parent(xvarsR[1])) , n0);
    monsAux := MonomialsOfDegree(Rx , g);

    listaExps := [];  listaMons := [];
    for m in monsAux do
        exps := Exponents(m);
        Append(~listaExps , exps);
        Append(~listaMons , &*[ xvarsR[t]^exps[t] : t in [1..n0] ]);
    end for;
    return listaExps , listaMons;
end function;

//////////////////////////////////////////////////////////////////////
//  CoeffList
//  --------------------------------------------------------------
//  Build the list of coefficient polynomials in the weights for every
//  monomial x^α of total degree dTot.  Returns
//      coeffs    – sequence of polynomials in the weights
//      expsList  – matching exponent vectors α.
//
//////////////////////////////////////////////////////////////////////
function CoeffList(n , d : K := RationalField())
K := RationalField();
    R , Ws , F := BuildNetworkDataNamed(n,d);

    n0 := n[1];
    dTot := 1;  for g in d do  dTot *:= g;  end for;

    numVars := #Names(R);
    numW := numVars - n0;                 // weights before x‑variables

    Rx<x> := PolynomialRing(K , n0);
    monsX := MonomialsOfDegree(Rx , dTot);
    expsList := [ Exponents(m) : m in monsX ];
    m := #expsList;

    // dictionary α ↦ index
    index := AssociativeArray();
    for i in [1..m] do  index[expsList[i]] := i;  end for;

    coeffs := [ R!0 : i in [1..m] ];

    for f in Eltseq(F) do
        monsF  := Monomials(f);
        cfsF   := Coefficients(f);

        for k in [1..#monsF] do
            mon  := monsF[k];
            c0   := cfsF[k];
            exps := Exponents(mon);

            xExps := exps[numW+1 .. numVars];
            if &+xExps eq dTot and IsDefined(index , xExps) then
                iPos := index[xExps];

                weightMon := R!1;
                for j in [1..numW] do
                    if exps[j] ne 0 then
                        weightMon *:= (R.j)^exps[j];
                    end if;
                end for;

                coeffs[iPos] +:= c0 * weightMon;
            end if;
        end for;
    end for;

    return coeffs , expsList;
end function;

//////////////////////////////////////////////////////////////////////
//  ParametrizationNamed
//  --------------------------------------------------------------
//  Build the projective coefficient map φ : A^{numW} → P^{m-1}
//  with readable weight names in the domain and w0,…,w_{m-1} in the
//  target.  Returns φ and its homogeneous coordinate list.
//
//////////////////////////////////////////////////////////////////////
function ParametrizationNamed(n , d : K := RationalField())
K := RationalField();
    coeffsR , _ := CoeffList(n,d);

    L := #n - 1;  numW := &+[ n[i+1]*n[i] : i in [1..L] ];

    weightNames := [ Sprint(Names(Parent(coeffsR[1]))[i]) : i in [1..numW] ];

    Dom := AffineSpace(K , numW);
    AssignNames(~Dom , weightNames);
    S := CoordinateRing(Dom);

    subst := [ S.i : i in [1..numW] ];
    n0 := n[1];  for j in [1..n0] do Append(~subst , K!0); end for;

    coeffsS := [ Evaluate(c , subst) : c in coeffsR ];

    m := #coeffsS;
    Cod := ProjectiveSpace(K , m-1);
    AssignNames(~Cod , [ Sprintf("w%o", i-1) : i in [1..m] ]);

    phi := map< Dom -> Cod | coeffsS >;
    return phi , coeffsS;
end function;

//////////////////////////////////////////////////////////////////////
//  AffineGaugeParametrizationNamed
//  
//  PURPOSE
//      • For every weight matrix W_i set *every entry in its last column*
//        (all rows, column index n_i-1) equal to 1  (gauge condition).
//      • De-homogenize the projective parametrization by dividing all
//        coordinates by the last one, yielding an affine map
//
//            φ_A :  A^{k}  →  A^{m-1},
//
//        where  k =  (total weights) − (number of fixed weights).
//      • Domain variables keep the readable weight names that remain
//        free; codomain coordinates are named u1,…
//
//  RETURNS   φ_A  and the list of its affine coordinate functions.
//
//////////////////////////////////////////////////////////////////////
function AffineGaugeParametrizationNamed(n , d)
K := RationalField();
    phiProj , coeffsFull := ParametrizationNamed(n,d);

    // -------- indices of weights to be fixed (all rows, last column) ---
    L := #n - 1;
    fixed := [];
    offset := 0;
    for i in [1..L] do
        rows := n[i+1];  cols := n[i];
        for r in [0..rows-1] do
            Append(~fixed , offset + r*cols + (cols - 1) + 1);
        end for;
        offset +:= rows*cols;
    end for;
    numW := offset;                              // total #weights
    kept := [ j : j in [1..numW] | j notin fixed ];
    k := #kept;                                  // free weights

    // -------- build new affine domain with readable names --------------
    keptNames := [ Sprint(Names(Domain(phiProj))[j]) : j in kept ];
    Dom := AffineSpace(K , k);
    AssignNames(~Dom , keptNames);
    SA := CoordinateRing(Dom);

    // -------- ring homomorphism  R → SA  (fixed weights ↦ 1) -----------
    mp := [];  ptr := 1;
    for j in [1..numW] do
        Append(~mp , (j in fixed) select SA!1 else SA.ptr);
        if j notin fixed then ptr +:= 1; end if;
    end for;
    Rsrc := Parent(coeffsFull[1]);
    phiRS := hom< Rsrc -> SA | mp >;

    polGauge := [ phiRS(f) : f in coeffsFull ];

    // -------- de-homogenize (divide by last coordinate) ----------------
    firstPoly := polGauge[1];
    coordsA  := [ polGauge[i]/firstPoly : i in [2..#polGauge] ];

    // -------- affine codomain with names u1,…,u_{m-1} ------------------
    Cod := AffineSpace(K , #coordsA);
    AssignNames(~Cod , [ Sprintf("u%o", i) : i in [1..#coordsA] ]);

    phiAff := map< Dom -> Cod | coordsA >;
    return phiAff , coordsA;
end function;


//////////////////////////////////////////////////////////////////////
//  PrettyMapStrings
//  --------------------------------------------------------------
//  Convert each coordinate polynomial of a map into a readable string
//  by replacing $.k with the actual weight name; returns a sequence
//  of strings without any “uᵢ = ” prefix.
//
//////////////////////////////////////////////////////////////////////
function PrettyMapStrings(phi)
    domNames := Names(Domain(phi));
    polys    := DefiningPolynomials(phi);

    out := [];
    for f in polys do
        s := Sprint(f);
        for k in [#domNames .. 1 by -1] do
            s := SubstituteString(s , Sprintf("$.%o",k) , domNames[k]);
        end for;
        Append(~out , s);
    end for;
    return out;
end function;

//////////////////////////////////////////////////////////////////////
//  PrettyMapPolys
//  --------------------------------------------------------------
//  Same replacement as PrettyMapStrings but returns the expressions as
//  actual polynomials / rational functions in the coordinate ring.
//
//////////////////////////////////////////////////////////////////////
function PrettyMapPolys(phi)
    domNames := Names(Domain(phi));
    polys    := DefiningPolynomials(phi);
    Rdom     := CoordinateRing(Domain(phi));

    subst := [];
    for k in [1..#domNames] do
        subst[k] := Rdom.k;
    end for;

    out := [];
    for f in polys do
        Append(~out , Evaluate(f , subst));
    end for;
    return out;
end function;


//////////////////////////////////////////////////////////////////////
//  GaugeJacobianRankRandom
//  --------------------------------------------------------------
//  Compute the Jacobian of the affine, gauged map φ_A, evaluate it at a
//  random integer point P ∈ [-5,5]^k (retry ≤ 10 until denominators ≠ 0),
//  and return  < rank , P >.
//
//////////////////////////////////////////////////////////////////////
function GaugeJacobianRankRandom(n , d)
    K := RationalField();
    phiA , coords := AffineGaugeParametrizationNamed(n,d);

function JacobianRank(phiA, coords)
    A := Domain(phiA);
    k := BaseField(A);
    n := Dimension(A);
    m := #coords;
    
    // Handle zero-dimensional domain
    if n eq 0 then
        return 0;
    end if;
    
    // The coordinate functions are in a rational function field
    F := Parent(coords[1]);
    
    // Construct the Jacobian matrix by differentiating with respect to variable indices
    J := ZeroMatrix(F, m, n);
    for i in [1..m] do
        for j in [1..n] do
            J[i,j] := Derivative(coords[i], j);  // j is the variable index
        end for;
    end for;
    
    // Attempt to evaluate at a random point (avoid division by zero)
    attempts := 0;
    max_attempts := 100;
    success := false;
    while attempts lt max_attempts and not success do
        // Generate random point in affine space
        pt := [Random(k,100) : i in [1..n]];
        attempts +:= 1;
        
        try
            // Evaluate all entries of the Jacobian matrix at pt
            Jval := Matrix(k, m, n, [Evaluate(entry, pt) : entry in Eltseq(J)]);
            success := true;
        catch e
            // Evaluation failed (likely due to division by zero), retry
            continue;
        end try;
    end while;
    
    if not success then
        error "Could not evaluate Jacobian at a random point after 100 attempts";
    end if;
    return Rank(Jval),pt;
end function;

JvalR,pt := JacobianRank(phiA, coords);
return JvalR,pt;
end function;


//////////////////////////////////////////////////////////////////////
//  NeuroVarietyStats
//  --------------------------------------------------------------
//  Compute key dimensions for the neuro‑variety:
//
//    expdim    – expected dimension (min{k , N})
//    dimActual – generic Jacobian rank (over ‘tries’ random points)
//    fiberDim  – k − dimActual (generic fibre dimension)
//    defective – true iff dimActual < expdim
//    bestP     – point achieving dimActual
//
//  RETURNS  < expdim , dimActual , fiberDim , defective , bestP >
//
//////////////////////////////////////////////////////////////////////
function NeuroVarietyStats(n , d : K := RationalField() , tries := 10)
    // expected dimension
    L := #n - 1;
    k    := &+[ n[i+1]*(n[i]-1) : i in [1..L] ];

    dTot := 1;  for g in d do  dTot *:= g;  end for;
    n0 := n[1];   nL := n[#n];
    N  := nL*Binomial(n0 - 1 + dTot , n0 - 1) - nL;
    T := Binomial(n[L-1] - 1 + d[L-1] , n[L-1]-1)-1;
    h := &+[ n[i+1]*(n[i]-1) : i in [1..L-2] ] + T;
    expdim := Minimum(Minimum(k , N),h);

    // generic Jacobian rank
    dimActual := 0;   bestP := [];
    for t in [1..tries] do
        rank,P := GaugeJacobianRankRandom(n , d);
        if rank gt dimActual then
            dimActual := rank;   bestP := P;
        end if;
        if dimActual eq expdim then break; end if;
    end for;

    fiberDim  := k - dimActual;
    defective := dimActual lt expdim;

    return < expdim , dimActual , fiberDim , defective , bestP >;
end function;

//////////////////////////////////////////////////////////////////////
//  Example usage 
//////////////////////////////////////////////////////////////////////

SetColumns(1000);

n := [2,2,3,1];
d := [3,3];

phi , coeffs := ParametrizationNamed(n , d);
phiAff , _ := AffineGaugeParametrizationNamed(n , d);

exprStr := PrettyMapStrings(phiAff);

stats := NeuroVarietyStats(n , d : tries := 7);
expdim , dimActual , fiberDim , defective , bestP := Explode(stats);

print "expdim    =", expdim;
print "dimActual =", dimActual;
print "fiberDim  =", fiberDim;
print "defective =", defective;

//////////////////////////////////////////////////////////////////////
//  Pretty-Print Map Equations
//  --------------------------------------------------------------
//  PURPOSE
//      To format and display the defining equations of the map `phiAff`
//      in a clean, human-readable, line-by-line format.
//
//  PROCESS
//      For each equation string `s` and corresponding output variable
//      name `v`, this block prints the formatted line:
//
//          v = s
//
//      where   `s`  is an element of  PrettyMapStrings(phiAff)
//      and     `v`  is an element of  Names(Codomain(phiAff))
//
//////////////////////////////////////////////////////////////////////
prettyEquations := PrettyMapStrings(phiAff);
codNames := Names(Codomain(phiAff));
for i in [1..#prettyEquations] do
    printf "%o = %o\n", codNames[i], prettyEquations[i];
end for;

///////////////////////////////////////////////////////////////////////////
//  ExhaustiveSmallDefectTest  –  single-layer front-end for quick search
//
//         expd - N < 0
//  with   N = Binomial( n0-1 + d1·…·d_{L-1} , n0-1 ) − 1
///////////////////////////////////////////////////////////////////////////

/* USER CONSTANTS – unchanged */
maxWidth     := 5;
maxLastWidth := 1;
maxDepth     := 4;
minExpo      := 2;
maxExpo      := 5;
tries        := 50;

/* generate all integer tuples */
procedure GatherTuples(len , lo , hi , tmp , ~bag)
    if #tmp eq len then
        Append(~bag , tmp);  return;
    end if;
    for v in [lo..hi] do
        GatherTuples(len , lo , hi , tmp cat [v] , ~bag);
    end for;
end procedure;

/* header */
print "n-seq            d-seq          exp  act  fib  defective";

for L in [3 .. maxDepth] do

    /* width lists  n = [n0 … nL] */
    nTuples := [];  GatherTuples(L+1 , 1 , maxWidth , [] , ~nTuples);

    for n in nTuples do

        /* width filters */
        if n[1] lt 2                        then continue; end if;
        if &or[ n[i] lt 2 : i in [2..L] ]   then continue; end if;
        if n[L+1] lt 1                      then continue; end if;
        if n[L+1] gt maxLastWidth           then continue; end if;

        /* exponent lists  d = [d1 … d_{L-1}] */
        dTuples := [];  GatherTuples(L-1 , minExpo , maxExpo , [] , ~dTuples);

        for d in dTuples do

            /* binomial inequality  ∀ i = 0 … L-2 */
            ok := true;
            for i in [1 .. L-1] do
                if Binomial(n[i]-1 + d[i] , n[i]-1) le n[i+1] and Binomial(n[i]-1 + d[i] , n[i]-1) le n[i+1]*(n[i]-1)+n[i+1]  then
                    ok := false; break;
                end if;
            end for;
            if not ok then continue; end if;

            /* compute stats (user routine) */
            stats := NeuroVarietyStats(n , d : tries := tries);
            expd , act , fib , defect , _ := Explode(stats);

            /* -------- NEW FILTER  expd - N < 0 --------------------- */
            prodD := &* d;                                // d1·…·d_{L-1}
            N := Binomial( n[1]-1 + prodD , n[1]-1 ) - 1; // new ambient dim
            if  act gt N then continue; end if;
            /* ------------------------------------------------------- */

            printf "%o  %o   %2o  %2o  %2o %2o %o\n",
                   n , d , expd , act , fib , N,
                   defect select "YES" else "NO";
        end for;
    end for;
end for;

///////////////////////////////////////////////////////////////////////////
//  ExhaustiveSmallDefectTest  –  single-layer front-end for quick search
///////////////////////////////////////////////////////////////////////////
//
//  For every width list  n = [n0,…,nL]  and exponent list  d = [d1,…,d_{L-1}]
//
//      • width bounds      1 ≤ n_i ≤ maxWidth      (except final ≤ maxLastWidth)
//      • depth             2 ≤ L ≤ maxDepth
//      • exponent bounds   minExpo ≤ d_i ≤ maxExpo
//      • input / output    n0 ≥ 2 , n_L ≥ 1
//      • all hidden widths n1 … n_{L-2} ≥ 2
//      • binomial test     binom(n_i-1+d_{i+1}, n_i-1)  > n_{i+1}
//
//  It prints:  expected dim, actual dim, fibre dim, defect flag.
//
///////////////////////////////////////////////////////////////////////////


//////////////////////////// USER CONSTANTS ///////////////////////////////
maxWidth     := 5;    // upper bound for every width n_i
maxLastWidth := 1;    // specific upper bound for n_L
maxDepth     := 4;    // test depths L = 2 … maxDepth
minExpo      := 2;    // exponent lower bound
maxExpo      := 5;    // exponent upper bound
tries        := 50;   // Jacobian-rank samples for NeuroVarietyStats
///////////////////////////////////////////////////////////////////////////


/////////////////// utility: generate integer tuples //////////////////////
procedure GatherTuples(len , lo , hi , tmp , ~bag)
    if #tmp eq len then
        Append(~bag , tmp);  return;
    end if;
    for v in [lo..hi] do
        GatherTuples(len , lo , hi , tmp cat [v] , ~bag);
    end for;
end procedure;


///////////////////////////////// HEADER //////////////////////////////////
print "n-seq            d-seq          exp  act  fib  defective";


//////////////////// main double-loop over depths /////////////////////////
for L in [3..maxDepth] do

    // ---- all width lists  n = [n0,…,nL] ------------------------------
    nTuples := [];  GatherTuples(L+1 , 1 , maxWidth , [] , ~nTuples);

    for n in nTuples do

        /* width filter */
        if n[1] lt 2                    then continue; end if; // n0 ≥ 2
        if &or[ n[i] lt 2 : i in [2..L] ] then continue; end if; // n1…n_{L-1} ≥ 2
        if n[L+1] lt 1                  then continue; end if; // n_L ≥ 1
        if n[L+1] gt maxLastWidth       then continue; end if; // n_L ≤ bound

        // ---- all exponent lists  d = [d1,…,d_{L-1}] -----------------
        dTuples := [];  GatherTuples(L-1 , minExpo , maxExpo , [] , ~dTuples);

        for d in dTuples do

            /* binomial inequality  ∀ i = 0 … L-2 */
            bad := false;
            for i in [1 .. L-1] do                     // Magma indices 1-based
                lhs := Binomial( n[i]-1 + d[i] , n[i]-1 );
                if lhs le n[i+1] then bad := true; break; end if;
            end for;
            if bad then continue; end if;

            /* compute stats (user's own routine) */
            stats := NeuroVarietyStats(n , d : tries := tries);
            expd , dim , fib , defect , _ := Explode(stats);

            printf "%o  %o   %2o  %2o  %2o   %o\n",
                   n , d , expd , dim , fib ,
                   defect select "YES" else "NO";
        end for;
    end for;
end for;



//////////////////////////////////////////////////////////////////////
//  ExhaustiveSmallDefectTest (Secant varieties of Veronese varieties i.e. L = 2, n_L = 1)
//
//  PURPOSE
//      Iterate over every network satisfying
//         • widths      n_i ∈ [1 .. maxWidth]
//         • output      n_L ≤ maxLastWidth
//         • depth       L   ∈ [2 .. maxDepth]
//         • exponents   d_i ∈ [minExpo .. maxExpo]
//         • constraints n0 ≥ 2 , n_L ≥ 1
//      For each pair (n,d) print: expected dimension, actual dimension,
//      generic fibre dimension, and defect flag.
//
//  USER CONSTANTS
//      maxWidth      – upper bound for every n_i
//      maxLastWidth  – upper bound specifically for the last width n_L
//      maxDepth      – largest depth L  (layers = L+1)
//      minExpo       – lower bound   for every d_i
//      maxExpo       – upper bound   for every d_i
//      tries         – Jacobian‑rank samples per architecture
//
//////////////////////////////////////////////////////////////////////

maxWidth     := 7;   // any n_i  ≤ 7
maxLastWidth := 1;   // restrict n_L ≤ 1    
maxDepth     := 2;   // up to L = 2
minExpo      := 3;   // d_i ≥3
maxExpo      := 5;   // d_i ≤ 7
tries        := 50;  // Jacobian attempts

// helper: collect all integer tuples of length len with entries in [lo..hi]
procedure GatherTuples(len , lo , hi , tmp , ~bag)
    if #tmp eq len then
        Append(~bag , tmp);  return;
    end if;
    for v in [lo..hi] do
        GatherTuples(len , lo , hi , tmp cat [v] , ~bag);
    end for;
end procedure;

print "n‑seq            d‑seq          exp  act  fib  defective";

for L in [2..maxDepth] do

    // ----- all width sequences  n = [n0 … nL] -------------------------
    nSeq := [];  GatherTuples(L+1 , 1 , maxWidth , [] , ~nSeq);

    for n in nSeq do
        if n[1] lt 2              then continue; end if;   // need ≥ 2 inputs
        if n[#n] lt 1             then continue; end if;   // need ≥ 1 output
        if n[#n] gt maxLastWidth  then continue; end if;   // respect new bound

        // ----- all exponent sequences  d = [d1 … d_{L-1}] -------------
        dSeq := [];  GatherTuples(L-1 , minExpo , maxExpo , [] , ~dSeq);

        for d in dSeq do
            stats := NeuroVarietyStats(n , d : tries := tries);
            expd , dim , fib , defect , _ := Explode(stats);

            print Sprintf("%o  %o   %2o  %2o  %2o   %o",
                          n , d , expd , dim , fib ,
                          defect select "YES" else "NO");
        end for;
    end for;
end for;

///////////////////////////////////////////////////////////////////////////
//  monlex
//
//  Purpose :  Return every monomial of total degree d in the multivariate
//             polynomial ring R, listed in *ascending* lexicographic order.
//
//  Usage    :  mons := monlex(R, d);
//
//  Input    :  R  –  A multivariate polynomial ring (type RngMPol).
//              d  –  A non-negative integer giving the total degree
//                     of the monomials to be enumerated.
//
//  Output   :  A sequence (type SeqEnum[RngMPolElt]) containing all
//              monomials of degree d in R, ordered lexicographically
//              with respect to the variable order of R.
//
//  Example  :  R<x,y,z> := PolynomialRing(Rationals(), 3);
//              mons := monlex(R, 2);   // [ x^2, x*y, x*z, y^2, y*z, z^2 ]
///////////////////////////////////////////////////////////////////////////
monlex := function(R, d)
    Rlex, phi := ChangeOrder(R, "lex");
    S := [ phi(m) : m in MonomialsOfDegree(R, d) ];
    mons := Reverse(Sort(S));   // ascending lex
    return mons;
end function;

///////////////////////////////////////////////////////////////////////////
//  Composite‑Veronese and projection construction             //
///////////////////////////////////////////////////////////////////////////

// USER INPUT  (edit these three lines and run the whole file) 
n0      :=  1;                 // = n     (dimension of the first P^n)
degrees := [3, 3];       // = (d1,…,d_{L-1})
j       :=  2;                 // stage at which you want ν_{d1…dj}, proj…
K       := Rationals();       // or GF(p) etc.  (optional)

// 0.  Basic checks                                            
require (j ge 1) and (j le #degrees):
       "‖  j must be between 1 and #degrees = ", #degrees;
require n0 ge 0:
       "‖  n0 must be ≥ 0";

// 1.  Initial projective space                           
P0<x0>    := ProjectiveSpace(K, n0);
Rprev     := CoordinateRing(P0);
Pprev     := P0;

// Data containers so we can get back anything we like later
ProjSpaces  := [* P0 *];                // P^{N_{d1}}, P^{N_{d2}}, …
Embeddings  := [* *];                   // individual Veronese maps
Ndims       := [];                      // the N_{di}

// 2.  Iterated Veronese embeddings up to stage j       
for idx in [1..j] do
    dpi        := degrees[idx];
    mons       := monlex(Rprev, dpi);
    Ndi        := #mons;           Append(~Ndims, Ndi);

    // give each new ambient space distinct coordinate names y<i>_<m>
    coordNames := [ Sprintf("y%o_%o", idx, k) : k in [1..Ndi] ];
    PNdi<y>    := ProjectiveSpace(K, Ndi-1);
    AssignNames(~PNdi, coordNames);

    ver        := map<Pprev -> PNdi | mons>;

    Append(~ProjSpaces, PNdi);
    Append(~Embeddings,  ver);

    // advance to the next level
    Pprev := PNdi;
    Rprev := CoordinateRing(Pprev);
end for;

// 3.  Compose to get  ν_{d1…dj}                             
nuMap := &*[ Embeddings[k] : k in [1..#Embeddings] ];

// 4.  Compute linear equations of the image and the resulting projection                           
Ptarget  := ProjSpaces[#ProjSpaces];     // this is P^{N_{dj}}
Vimage   := Image(nuMap);
Jirr     := IrrelevantIdeal(Ptarget);

// (i)   all linear forms vanishing on the image
B1 := [ f : f in Basis(Ideal(Vimage) meet Jirr) | Degree(f) eq 1 ];
Rtar := CoordinateRing(Ptarget);

// turn each linear form into a point in the dual projective space
points := [ [ MonomialCoefficient(f, Rtar.l) : l in [1..Ndims[#Ndims]] ]
            : f in B1 ];
pts    := [ Cluster(Ptarget![c : c in pt]) : pt in points ];

// intersect all those clusters to obtain *the* linear space
U := pts[1];
for k in [2..#pts] do
    U := Scheme(Ambient(U), Ideal(U) meet Ideal(pts[k]));
end for;

// (ii)  independent linear generators of that space
Bc := [ f : f in Basis(Ideal(U) meet Jirr) | Degree(f) eq 1 ];
PP<[v]> := ProjectiveSpace(K, #Bc-1);

// the desired projection map
projMap := map<Ptarget -> PP | Bc>;

// 5.  OPTIONAL – print dimensions and make the maps visible  
print "• Composite Veronese  ν_{d1…dj} :  ", Domain(nuMap),
      " → ", Codomain(nuMap);
print "• Projection          proj_{d1…dj} : ", Domain(projMap),
      " → ", Codomain(projMap);
print "  (kernel dimension = ", Dimension(Domain(projMap)) -
                               Dimension(Codomain(projMap)), ")";

// 6.  Return the two maps 
nuMap, projMap;

///////////////////////////////////////////////////////////////////////////
// VerifyThmMainHypotheses (function version)
// Checks (i)–(iii) of Theorem th:main for (n,d).
// Returns a record with fields: ok, basic, cond1, cond2, cond3, and diagnostics.
///////////////////////////////////////////////////////////////////////////
function VerifyThmMainHypotheses(n , d)
    RF := recformat<
        ok        : BoolElt,
        basic     : BoolElt,   // length checks, L>=2, n_L>=2
        cond1     : BoolElt,   // (i) room inequalities
        cond2     : BoolElt,   // (ii) AH non-defectiveness at last level
        cond3     : BoolElt,   // (iii) expdim((...,1)) = parameter count
        L         : RngIntElt,
        n0        : RngIntElt,
        nLm2      : RngIntElt, // n_{L-2}
        nLm1      : RngIntElt, // n_{L-1}
        nL        : RngIntElt, // n_{L}
        dLm1      : RngIntElt, // d_{L-1}
        k_n1      : RngIntElt, // parameter count for (n0,...,n_{L-1},1)
        N_n1      : RngIntElt, // ambient affine dim for (..,1)
        h_n1      : RngIntElt, // improved cap term
        exp_n1    : RngIntElt  // expected dim for (..,1)
    >;

    R := rec< RF | ok := false, basic := false, cond1 := false, cond2 := false, cond3 := false,
                    L := 0, n0 := 0, nLm2 := 0, nLm1 := 0, nL := 0, dLm1 := 0,
                    k_n1 := 0, N_n1 := 0, h_n1 := 0, exp_n1 := 0 >;

    // --- basic shape checks ---
    if #n lt 3 then return R; end if;
    L := #n - 1;
    if #d ne L-1 then return R; end if;
    if L lt 2 or n[#n] lt 2 then return R; end if;

    R`L  := L;
    R`n0 := n[1];
    R`nL := n[#n];

    // (i) room inequalities: n_{i-1}+n_i-1 < C(n_{i-1}-1 + d_i, n_{i-1}-1), for i=1..L-1
    cond1 := true;
    for i in [1..L-1] do
        lhs := n[i] + n[i+1] - 1;
        rhs := Binomial(n[i]-1 + d[i], n[i]-1);
        if not (lhs lt rhs) then cond1 := false; break; end if;
    end for;

    // (ii) Non-defectiveness for V^{n_{L-2}-1}_{d_{L-1}} at r = n_{L-1}
    // Alexander–Hirschowitz exceptions:
    //  - d=2, vars>=3, and 2 <= r <= vars-1
    //  - (vars,d,r) in {(3,4,5), (4,4,9), (5,3,8)}
    vars := n[L-1];    // n_{L-2}
    deg  := d[L-1];    // d_{L-1}
    r    := n[L];      // n_{L-1}
    R`nLm2 := vars;  R`nLm1 := r;  R`dLm1 := deg;

    isExc := false;
    if deg eq 2 and vars ge 3 and 2 le r and r le vars-1 then
        isExc := true;
    end if;
    if <vars,deg,r> eq <3,4,5> or <vars,deg,r> eq <4,4,9> or <vars,deg,r> eq <5,3,8> then
        isExc := true;
    end if;
    cond2 := not isExc;

    // (iii) For (n0,...,n_{L-1},1), improved expected dimension equals parameter count
    ExpectedDim := function(nlist, dlist)
        local LL, k, dTot, N, T, h, i;
        LL := #nlist - 1;
        k := &+[ nlist[i+1]*(nlist[i]-1) : i in [1..LL] ];
        dTot := 1; for i in [1..#dlist] do dTot *:= dlist[i]; end for;
        N := nlist[#nlist]*Binomial(nlist[1]-1 + dTot, nlist[1]-1) - nlist[#nlist];
        T := Binomial( nlist[LL-1] - 1 + dlist[LL-1], nlist[LL-1]-1 ) - 1; // last Veronese ambient dim
        if LL ge 2 then
            h := (&+[ nlist[i+1]*(nlist[i]-1) : i in [1..LL-2] ]) + T;
        else
            h := T;
        end if;
        return Minimum(Minimum(k,N), h), k, N, h;
    end function;

    n1 := n;  n1[#n1] := 1;  // replace last width by 1
    exp_n1, k_n1, N_n1, h_n1 := ExpectedDim(n1, d);
    cond3 := (exp_n1 eq k_n1);

    // Fill and return
    R`basic   := true;
    R`cond1   := cond1;
    R`cond2   := cond2;
    R`cond3   := cond3;
    R`k_n1    := k_n1;
    R`N_n1    := N_n1;
    R`h_n1    := h_n1;
    R`exp_n1  := exp_n1;
    R`ok      := R`basic and R`cond1 and R`cond2 and R`cond3;
    return R;
end function;

///////////////////////////////////////////////////////////////////////////
R := VerifyThmMainHypotheses([3,4,5,2], [3,5]);
print R`ok, R`cond1, R`cond2, R`cond3;
