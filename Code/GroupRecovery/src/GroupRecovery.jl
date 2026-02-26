module GroupRecovery

using Oscar
using Plots
using Distributions
import Oscar: PermGroup, PermGroupElem, symmetric_group, rand, degree
using SpecialFunctions: logabsbinomial


export
############### MAIN STRUCT & FUNCTIONS ##########
    RecoveryDatum,
    sample,
    degree,
    samples_used,
    reset_sample_count!,
    reset_P!,
    reset_Q!,
    reset_all!,
    update_p_bound,
    update_order_bound,
    update_B_bound,
    update_all_bounds, 
    add_permutation_property,
    add_group_property,
    NaiveGroupRecovery,
    Q_ErrorDetection,
    NiAGRA,
    main_group_recovery,
    ############### STATISTICAL TESTS & METHODS ###########
    success_rate_check,
    ############## MAIN TESTS ##############
    giant_test,
    subgroup_test,
    alternating_test,
    ktransitivity_test,
    transitivity_test,
    single_orbit_recovery,
    orbit_recovery,
    orbit_coarsening_test,
    single_orbit_confirmation,
    orbit_confirmation,
    find_supergroup,
    ############## helper functions ##########
    is_giant,
    giant_pair,
    lower_if_giant,
    upper_if_nongiant,
    subgroup_lower_if_high,
    subgroup_upper_if_low,
    fixed_points,
    fixed_ktuples,
    fixed_pts_sample,
    fixed_ktuples_sample,
    orbit_agreement,
    young_subgroup,
    single_orbit_confirmation2,
    add_subgroup_permutation_property,
    smallest_divisor,
    B, 
    pak_bound,
    qp_bound,
    gamma_bound,
    HoeffdingSampleSize,
    HoeffdingDistinguisher,
    sample_has_property,
    estimate_error_from_orbits,
    binomial_confidence_interval, 
    weylgroup,
    tally, 
    visualize_findspg,
    categorize_supergroups,
    perm_in_group,
    is_subgroup_of




######################
const verbose = true
############## STRUCTS ##############

""" RecoveryDatum

The main struct for holding data relevant to recovering an unknown permutation group G from an error-prone sampling function X.  

Fields: 
- X::Function: An error-prone sampling function that when called returns a PermGroupElem sampled from G with probability 1-p and from Sₙ with probability p.
- P::Vector{Function}: A collection of permutation properties that hold for every g ∈ G. Each property is represented as a function that takes in a PermGroupElem and returns true or false.
- Q::Vector{Function}: A collection of group properties of G. Each property is represented as a function that takes in a PermGroup and returns true or false.
- n::Int64: The degree of the permutation group being sampled in X.
- p_bound::Float64: An upper bound on the error probability p of the sampling function X.
- order_bound::BigInt: An upper bound on the order of the group G.
- B_bound::Float64: An upper bound on the size of the set |S_P|/n!, where S_P is the set of permutations satisfying all properties in P.
- n_samples::Int64: The current number of samples used.

"""
Base.@kwdef mutable struct RecoveryDatum
    X::Function
    P::Vector{Function}=[] 
    Q::Vector{Function}=[]  
    n::Int64 
    p_bound::Float64 
    order_bound::BigInt
    B_bound:: Float64   
    n_samples::Int64 = 0 
    min_supergroup::PermGroup
end



"""
    RecoveryDatum(G::PermGroup, p::Float64; kwargs...)

Creates a RecoveryDatum for the permutation group G with a sampling error probability p. Additional keyword arguments can be provided to set the initial bounds and properties.

The default bound on p is 0.35.

"""
function RecoveryDatum(G::PermGroup, p::Float64; kwargs...)
    n = Oscar.degree(G)
    return RecoveryDatum(
        X = sample_with_error(G, p),
        n = n,
        p_bound = 0.35,
       # M_bound = Int(ceil(log2(factorial(big(n))))),
        order_bound = factorial(big(n)), #maybe better to represent with a group?
        B_bound = 1,
        min_supergroup = symmetric_group(n),
        kwargs...
    )
end

Base.copy(RD::RecoveryDatum)=RecoveryDatum(X=RD.X, n=(RD.n)*(RD.n-1), p_bound=RD.p_bound, M_bound=RD.M_bound, order_bound= factorial(big((RD.n)*(RD.n-1))), B_bound= RD.B_bound, n_samples=RD.n_samples, min_supergroup=RD.min_supergroup)

"""
    sample_with_error(G::PermGroup, p::Float64)

Returns a function that, when called, returns an element of G with probability 1-p and a random element of Sₙ with probability p.
"""
sample_with_error(G::PermGroup, p::Float64) = () -> rand() < p ? rand(symmetric_group(Oscar.degree(G))) : rand(G)


############## BASIC METHODS ##############

"""
    degree(RD::RecoveryDatum)
Returns the degree of the permutation group being sampled by the recovery datum RD.
"""
degree(RD::RecoveryDatum) = RD.n

"""
    samples_used(RD::RecoveryDatum)
Returns the number of samples used in the recovery datum RD.
"""
samples_used(RD::RecoveryDatum) = RD.n_samples


""" 
    reset_sample_count!(RD::RecoveryDatum)     
Resets the sample count of the recovery datum RD to zero.
"""
reset_sample_count!(RD::RecoveryDatum) = RD.n_samples=0

""" 
    reset_P!(RD::RecoveryDatum)
Resets the permutation properties of the recovery datum RD.
"""
reset_P!(RD::RecoveryDatum) = empty!(RD.P)

""" 
    reset_Q!(RD::RecoveryDatum)
Resets the group properties of the recovery datum RD.
"""
reset_Q!(RD::RecoveryDatum) = empty!(RD.Q)

""" 
    reset_all!(RD::RecoveryDatum)
Resets all properties, sample count, and bounds of the recovery datum RD.    
"""
function reset_all!(RD::RecoveryDatum)
    reset_P!(RD)
    reset_Q!(RD)
    reset_sample_count!(RD)
    RD.p_bound = 0.35
    #RD.M_bound = Int(ceil(log2(factorial(big(RD.n)))))
    RD.order_bound = factorial(big(RD.n))
    RD.B_bound = 1.0
end


update_B_bound(RD::RecoveryDatum, new_bound::Float64) = RD.B_bound = min(RD.B_bound, new_bound)

update_order_bound(RD::RecoveryDatum, new_bound::BigInt) = RD.order_bound = min(RD.order_bound, BigInt(new_bound))

update_p_bound(RD::RecoveryDatum, new_bound::Float64) = RD.p_bound = min(RD.p_bound, new_bound)

#update_M_bound(RD::RecoveryDatum, new_bound::Float64) = RD.M_bound = min(RD.M_bound, new_bound)

""" 
    update_all_bounds(RD::RecoveryDatum; p_bound = nothing, order_bound = nothing, B_bound = nothing)
    
Updates the bounds of the recovery datum RD. If a bound is not provided, it remains unchanged.
"""
function update_all_bounds(RD::RecoveryDatum; p_bound = nothing, order_bound = nothing, B_bound = nothing, M_bound = nothing)
    if p_bound !==nothing
        update_p_bound(RD, p_bound)
    end
    if order_bound !==nothing
        update_order_bound(RD, order_bound)
    end
    if B_bound !==nothing
        update_B_bound(RD, B_bound)
    end
    #if M_bound !==nothing
    #    #update_M_bound(RD, M_bound)
    #end 
end


"""
    add_group_property(RD::RecoveryDatum, f::Function)
Adds a new group property f to the recovery datum RD. The function f should take in a PermGroup and return true or false.
"""
add_group_property(RD::RecoveryDatum, f::Function) = (!(f in RD.Q)) ? push!(RD.Q, f) : nothing



"""
    add_permutation_property(RD::RecoveryDatum, f::Function)
Adds a new permutation property f to the recovery datum RD. The function f should take in a PermGroupElem and return true or false.
"""
add_permutation_property(RD::RecoveryDatum, f::Function) = (!(f in RD.P)) ? push!(RD.P, f) : nothing




""" sample(RD::RecoveryDatum)

Uses RD.X to return a PermGroupElem sample that satisfies all available permutation properties.
"""
function sample(RD::RecoveryDatum)
    for _=1:1000
        g = RD.X()
        RD.n_samples+=1
        if all(p(g) for p in RD.P)
            return g
        end
    end
    return error("Could not sample a permutation satisfying all properties in RD.P after 1000 attempts.")
end


"""    sample_has_property(RD::RecoveryDatum, f::Function)
Samples from the recovery datum RD and checks if the sampled permutation satisfies the property f, which should take in a PermGroupElem and return true or false.
"""
function sample_has_property(RD::RecoveryDatum, f::Function)
    g = sample(RD)
    #should this be g = RD.X() instead????
    return f(g)
end

function sample_has_property(g::PermGroupElem, f::Function)
    return f(g)
end


##################### PERMUTATION & GROUP PROPERTIES #####################


#### GROUP PROPERTIES ########
function is_giant(G::PermGroup)
    if is_transitive(G) && (is_natural_alternating_group(G) || is_natural_symmetric_group(G))
        return true
    else
        return false
    end
end

"""
    is_subgroup_of(H::PermGroup)
Returns a function that takes in a PermGroup G and checks if G is a subgroup of H.
"""
function is_subgroup_of(H::PermGroup)
    return (G::PermGroup)-> is_subset(G, H)
end


########## PERM PROPERTIES ##########

""" 
    perm_in_group(H::PermGroup)
Returns a function that takes in a PermGroupElem g and checks if g is an element of H.
"""
function perm_in_group(H::PermGroup)
    return (g::PermGroupElem) -> g in H
end


################### STATISTICAL TESTS & BOUNDS ####################

# lower bound on φ_k(G) given an upper bound M on ceil(log2(|G|)). If conservative=true, use the conservative bound from Pak's paper (this is the bound used in the RecoveringPermutationGroups paper). If conservative=false, then use the tighter bound that is equal to φ_k(Z_2^M)
function pak_bound(M::Int64, k; conservative=false)
    if conservative
        @assert k>M+3 "k is too low, must be at least $(M+4)"
        return 1-8/(2^(k-M))
        #or tighter bound is 
        #return 1-4/(2^(k-M+1)-1)
        #pg 9 of Igor Pak paper
    else
        @assert k>M-1 "k is too low, must be at least $(M)"
        return prod([1-(1/2^i) for i=(k-M+1):k] ) 
    end
end

# lower bound on φ_k(G) 
function pak_bound(G::PermGroup, k; conservative=false)
    M=Int(ceil(log2(order(G))))
    return pak_bound(M, k, conservative=conservative)
end


# upper bound on qₚ
function qp_bound(ptilde, B)
    return 2*ptilde/(B+1/B)
end


"""
    success_rate_check(RD::RecoveryDatum, k::Int64; conservative=false)

Computes a lower bound on γ(G,p,k) the probability that k samples from the recovery datum RD generate G. Corresponds to γ_{P,Q}(G,p,k).
"""
function success_rate_check(RD::RecoveryDatum, k::Int64; conservative=false)
    p = RD.p_bound
    B = RD.B_bound
    Mtilde = Int(ceil(log2(RD.order_bound)))
    verbose && println("p = $p, B = $B, Mtilde = $Mtilde")
    #verbose && println("Lower bound on pulling $k good samples:",(1-qp_bound(p,B))^k)
    #verbose && println("Lower bound on generating group from $k good samples based on order bound:", pak_bound(M,k,conservative=conservative))
    gamma_bound = success_rate_check(p, Mtilde, B, k, conservative=conservative)
    print(gamma_bound)
    if gamma_bound < 0
        return 0.0
    else
        return gamma_bound
    end
end


function success_rate_check(p::Float64, Mtilde::Int64, B::Float64, k::Int64; conservative = false)
    return (( (1-qp_bound(p,B)) )^k)*(pak_bound(Mtilde, k, conservative = conservative))
end



#=

#lower bound on γ(G,p,k)
function gamma_bound(G,p,k; conservative=false, property=false)
    if property==true
        p=p*(1-1/degree(G))
    end
    return ((1-p)^k)*(pak_bound(G, k, conservative=conservative))
end




#upper bound on p for gamma(G,p,k)>1/2
function ω(G,r; flag=true)
    n=degree(G)
    M=Int(ceil(log2(factorial(n-1))))
    k=M+r
    if flag==true #conservative
        return 1-(2^(r-3)/(2^(r-2) -1))^(1/k)
    else
        return 1-(1/(2*pak_bound(M,k, conservative=false)))^(1/k)
    end
end

#upper bound on p for gammaP(G,p,k)>1/2
function ω_property(G, B, r; flag=true)
    return (B+1/B)*ω(G,r;flag=flag)/2
end


=#


"""
    HoeffdingSampleSize(alpha, delta; range=1)
    HoeffdingSampleSize(alpha, ub, lb, min_rv_val, max_rv_val) 

Calculates the sample size N required to distinguish between two distributions with means differing by at least 2*delta, with confidence level 1-alpha, using Hoeffding's inequality. The random variable is assumed to be bounded within an interval of length 'range'.

Alternatively, the second method takes upper and lower bounds on the means (ub and lb) and the minimum and maximum possible values of the random variable to compute the range.
"""
HoeffdingSampleSize(alpha, delta; range=1) = ceil(Int, -log(alpha)*range^2 / (2*(delta^2)) )

HoeffdingSampleSize(alpha, ub, lb, min_rv_val, max_rv_val) = HoeffdingSampleSize(alpha, (lb - ub)/2, range = max_rv_val - min_rv_val)



"""
    HoeffdingDistinguisher(f, upper_for_low, lower_for_high, alpha; delta, min_rv_val, max_rv_val)

A mean-threshold distinguisher that uses Hoeffding's inequality to distinguish between two cases: one where the expected value of f() is at most upper_for_low, and one where it is at least lower_for_high. It returns :low and :high for each respective case, with a 1-alpha confidence level that the sample mean is within delta of the true mean. The random variable f() is assumed to be bounded between min_rv_val and max_rv_val. 
"""
function HoeffdingDistinguisher(
    f::Function, 
    upper_for_low::Float64, 
    lower_for_high::Float64, 
    alpha::Float64; 
    delta=(lower_for_high - upper_for_low)/2, 
    min_rv_val=0, 
    max_rv_val=1,
    boolean_valued=false)

    verbose=true
    @assert upper_for_low < lower_for_high "upper bound for low category must be less than lower bound for high category"
    @assert 0 < alpha < 1 "alpha must be in (0,1)"
    verbose && println("Using Hoeffding distinguisher with confidence level $(1-alpha)")
    #verbose && println("delta = $delta, min_rv_val = $min_rv_val, max_rv_val = $max_rv_val")
    
    range = max_rv_val - min_rv_val
    #=
    if boolean_valued
        range=1
        orig_f=f
        f=()->orig_f() ? 1 : 0
    end
    =#
    N = HoeffdingSampleSize(alpha, delta, range=range)
    c = 0

    for _ in 1:N
        c+=f()
    end
    prop = c/N

    verbose && println("Sampled $N times, got proportion $prop")
    
    if prop >= (lower_for_high + upper_for_low)/2
        return :high
    else
        return :low
    end
end


# This function, rather than using the Hoeffding bound, will actually compute the exact probability of 
#   failing the distinguisher test when the sample size used is N, and will choose the first N for which
#   the confidence level is above 1-alpha.
"""
    explicit_distinguisher(f, upper_for_low, lower_for_high, alpha; min_rv_val=0, max_rv_val=1, boolean_valued=false)

A mean-threshold distinguisher that distinguishes between two cases: one where the expected value of f() is at most upper_for_low, and one where it is at least lower_for_high. It returns :low and :high for each respective case, with a confidence level of 1-alpha. This function computes the exact probability of failure based on the binomial distribution and chooses the sample size N accordingly, rather than relying on Hoeffding's inequality.

The function f() can return values in [min_rv_val, max_rv_val], which are normalized to [0,1] for the binomial test.
"""
function explicit_distinguisher(f, upper_for_low, lower_for_high, alpha; min_rv_val=0, max_rv_val=1, boolean_valued=false)
    @assert upper_for_low < lower_for_high "upper bound for low category must be less than lower bound for high category"
    @assert 0 < alpha < 1 "alpha must be in (0,1)"
    verbose && println("Using explicit distinguisher with alpha = $alpha")
    verbose && println("upper_for_low = $upper_for_low, lower_for_high = $lower_for_high")

    range = max_rv_val - min_rv_val
    
    #=
    # Normalize f() to return values in [0, 1]
    if boolean_valued
        orig_f = f
        f = () -> orig_f() ? 1 : 0
        # Bounds are already in [0, 1] for boolean case
    elseif range != 1 #if the range is not already [0, 1], we need to normalize =#
    if range != 1 #if the range is not already [0, 1], we need to normalize
        orig_f = f
        f = () -> (orig_f() - min_rv_val) / range
        # Normalize the bounds to [0, 1]
        upper_for_low = (upper_for_low - min_rv_val) / range
        lower_for_high = (lower_for_high - min_rv_val) / range
    end
    
    threshold = (lower_for_high + upper_for_low) / 2

    N = 5
    while true
        # Compute the probability of getting at least delta*N successes when the true success probability is p_low
        prob_fail_low = binomial_tail_upper(N, upper_for_low, ceil(Int, threshold * N))
            
        # Compute the probability of getting fewer than delta*N successes when the true success probability is p_high  
        prob_fail_high = binomial_tail_lower(N, lower_for_high, floor(Int, threshold * N - 1))
            
        total_prob_fail = prob_fail_low + prob_fail_high
        if total_prob_fail <= alpha
            break
        end
        N += 1
    end
    
    # Sample N times and compute normalized sum
    sum_normalized = 0.0
    for i in 1:N
        sum_normalized += f()
    end
    prop = sum_normalized / N
    verbose && println("Sampled $N times, got normalized proportion $prop")

    if prop >= threshold
        return :high
    else
        return :low
    end
end

# Compute P(X >= k_min) where X ~ Binomial(n, p) using log-space for numerical stability
function binomial_tail_upper(n, p, k_min)
    k_start = max(0, k_min)
    if k_start > n
        return 0.0
    end
    log_terms = [logabsbinomial(n, k)[1] + k * log(p) + (n - k) * log(1 - p) for k in k_start:n]
    return exp(logsumexp(log_terms))
end

# Compute P(X <= k_max) where X ~ Binomial(n, p) using log-space for numerical stability  
function binomial_tail_lower(n, p, k_max)
    k_end = min(n, k_max)
    if k_end < 0
        return 0.0
    end
    log_terms = [logabsbinomial(n, k)[1] + k * log(p) + (n - k) * log(1 - p) for k in 0:k_end]
    return exp(logsumexp(log_terms))
end


# Numerically stable log-sum-exp function
function logsumexp(x)
    if isempty(x)
        return -Inf
    end
    x_max = maximum(x)
    if isinf(x_max)
        return x_max
    end
    return x_max + log(sum(exp.(x .- x_max)))
end


"""
    SPRT_distinguisher(f, p0, p1; alpha=0.01, min_rv_val=0, max_rv_val=1, boolean_valued=false)

Sequential Probability Ratio Test to distinguish between:
- H0: E[f()] = p0 (low hypothesis)
- H1: E[f()] = p1 (high hypothesis)

Returns :low or :high when confident at error rate alpha (Type I and Type II).

The function f() can return values in [min_rv_val, max_rv_val], which are normalized to [0,1] for the binomial SPRT.
"""
function SPRT_distinguisher(f::Function, p0::Float64, p1::Float64, alpha=0.01; min_rv_val=0, max_rv_val=1, boolean_valued=false)
    @assert p0 < p1 "p0 must be less than p1"
    @assert 0 < alpha < 1 "alpha must be in (0,1)"
    
    range = max_rv_val - min_rv_val
    

    # Normalize f() to return values in [0, 1]
    #=if boolean_valued
        orig_f = f
        f = () -> orig_f() ? 1.0 : 0.0
        # Bounds are already in [0, 1] for boolean case
        p0_norm = p0
        p1_norm = p1
    else=#
    if range !=1
        # Normalize the function output to [0, 1]
        orig_f = f
        f = () -> (orig_f() - min_rv_val) / range
        # Normalize the bounds to [0, 1]
        p0 = (p0 - min_rv_val) / range
        p1 = (p1 - min_rv_val) / range
    end
    
    # Log likelihood ratio thresholds
    A = log((1-alpha)/alpha)
    B = log(alpha/(1-alpha))
    
    log_ratio = 0.0
    n = 0
    
    while true
        n += 1
        x = f()  # Normalized value in [0, 1]
        
        # Update log likelihood ratio for continuous [0,1] values
        # We model this as x ~ p*1 + (1-p)*0 = p, treating x as Bernoulli-like
        # For a normalized value, we compute likelihood ratio
        log_ratio += x * log(p1/p0) + (1-x) * log((1-p1)/(1-p0))
        
        verbose && (n % 100 == 0) && println("SPRT: n=$n, log_ratio=$log_ratio")
        
        # Check stopping conditions
        if log_ratio >= A
            verbose && println("SPRT stopped at n=$n, accepting H1 (:high)")
            return :high
        elseif log_ratio <= B
            verbose && println("SPRT stopped at n=$n, accepting H0 (:low)")
            return :low
        end
        
        # Safety check to prevent infinite loops
        if n > 100000
            @warn "SPRT did not terminate after 100000 samples, returning based on current evidence"
            return log_ratio > (A+B)/2 ? :high : :low
        end
    end
end



################## GIANT TEST ####################

#computes upper bound for p such that giant test can be used, based on degree of group

b(n) =(3*n^2 - n - 8.8 - sqrt(n^4 + 2*n^3 + 50.08*n^2 - 13.88*n - 199.584))/(4*n^2 - 15.74)


function generates_giant(S::Vector{PermGroupElem})
    return is_giant(permutation_group(degree(S[1]), S))
end

function giant_pair(RD::RecoveryDatum)
    return generates_giant([sample(RD), sample(RD)])
end

function giant_test_f(RD::RecoveryDatum)
    return () -> giant_pair(RD)
end


upper_if_nongiant(n,p) = 2*p*(1-p)+(1-1/n-0.93/n^2)*p^2

upper_if_nongiant(RD::RecoveryDatum) = upper_if_nongiant(degree(RD), RD.p_bound)

lower_if_giant(n,p) = (1-p+p^2)*(1-1/n-8.8/n^2)

lower_if_giant(RD::RecoveryDatum)=lower_if_giant(degree(RD), RD.p_bound)




""" 
    giant_test(RD::RecoveryDatum; alpha=0.01, method = :hoeffding, update = true)
Tests whether the unknown group G sampled by the recovery datum RD is a giant. Returns true if G is determined to be a giant, and false otherwise, with confidence level 1-alpha. The method argument specifies which statistical test to use: :hoeffding for the HoeffdingDistinguisher, :explicit for the explicit_distinguisher, and :sprt for the SPRT_distinguisher. The update argument specifies whether to update the properties and bounds of RD based on the test result.
"""
function giant_test(RD::RecoveryDatum; alpha=0.01, method = :hoeffding, update = true)
    n = degree(RD)
    @assert RD.p_bound < b(n) "Upper bound on error p is too high to perform giant test, given the group's degree. Must have RD.p_bound < $(b(n)). Current RD.p_bound = $(RD.p_bound)."

    ub = upper_if_nongiant(RD)
    lb = lower_if_giant(RD)
    category = nothing
    
    if method == :hoeffding
        category = HoeffdingDistinguisher(giant_test_f(RD), ub, lb, alpha)
    elseif method == :explicit
        category = explicit_distinguisher(giant_test_f(RD), ub, lb, alpha)
    elseif method == :sprt 
        category = SPRT_distinguisher(giant_test_f(RD), ub, lb, alpha)
    else
        error("Unknown method: $method. Use :hoeffding, :explicit, or :sprt")
    end


    if update == true
        if category == :high
            add_group_property(RD, is_giant)
            return true
        else
            add_group_property(RD, !is_giant)
            update_order_bound(RD, factorial(big(n-1)))
            #update_M_bound(RD, Int(ceil(log2(factorial(big(n-1))))))
            update_B_bound(RD, 1/n)
            return false
        end
    end

    return category == :high

end

################## SUBGROUP TEST ####################


function subgroup_lower_if_high(H::PermGroup, n::Int64, p::Float64)
    H_index = 1/(BigInt(index(symmetric_group(n),H)))
    return Float64((1-p)+p*H_index)
end

function subgroup_lower_if_high(RD::RecoveryDatum, H::PermGroup)
    subgroup_lower_if_high(H, degree(RD), RD.p_bound)
end


"""
    subgroup_test(RD::RecoveryDatum, H::PermGroup; alpha=0.01, method = :hoeffding)
Tests whether the unknown group G sampled by the recovery datum RD is a subgroup of H with confidence level at least 1-alpha. If the test concludes that G is a subgroup of H, it returns true; otherwise it returns false. If update the permutation properties and bounds of RD accordingly.    
"""
function subgroup_test(RD::RecoveryDatum, H::PermGroup; alpha=0.01, method = :hoeffding, update = false)
    ub = 0.5 
    lb = subgroup_lower_if_high(RD, H)
    perm_property = perm_in_group(H)
    subgroup_sample = () -> sample_has_property(RD, perm_property)
    category = nothing

    if method == :hoeffding
        category = HoeffdingDistinguisher(subgroup_sample, ub, lb, alpha)
    elseif method == :explicit
        category = explicit_distinguisher(subgroup_sample, ub, lb, alpha)
    else
        error("Unknown method: $method. Use :hoeffding or :explicit")
    end

    if category == :high
        if update == true
            add_permutation_property(RD, perm_property)
            add_group_property(RD, is_subgroup_of(H))
            if order(H) < RD.order_bound
                update_order_bound(RD, BigInt(order(H)))
                update_B_bound(RD, Float64(RD.order_bound)/factorial(big(degree(RD))))
            end
        end
        return true
    else
        return false
    end
end



#=
#add specific perm property that g ∈ H and also update bounds of rd
function add_subgroup_permutation_property(RD::RecoveryDatum, H::PermGroup)
    update_order_bound(RD, BigInt(order(H)))
    update_B_bound(RD, 1/Int( index(symmetric_group(degree(RD)), H ) ))
end
=#


################## ALTERNATING TEST ####################


function alternating_test(RD::RecoveryDatum; alpha=0.01, method = :hoeffding)
    p = RD.p_bound
    upper_for_low = (1-p)/2 + p/2
    lower_for_high = (1-p)+p/2
    category = nothing
    #perm_property = is_even

    if method == :hoeffding
        category = HoeffdingDistinguisher(() -> sample_has_property(RD, is_even), upper_for_low, lower_for_high, alpha)
    elseif method == :explicit
        category = explicit_distinguisher(() -> sample_has_property(RD, is_even), upper_for_low, lower_for_high, alpha)
    else
        error("Unknown method: $method. Use :hoeffding or :explicit")
    end
    if category == :high
        return true
    else
        return false
    end
end



################### TRANSITIVITY TESTS  ####################
#c= 1 + (1-p)*2 + p 

# expected vallue of Fix_k(X) if G is k-transitive is 1, while if G is not k-transitive, the expected value is at least (1-p)*2 +p = 2-p

function P(n,k) #n!/(n-k)!
    return prod([n:-1:n-k+1;])
end


function fixed_points(g::PermGroupElem, L::Vector{Int64})
    return count(i->g(i)==i, L)
end

function fixed_points(g::PermGroupElem)
    return fixed_points(g, [1:degree(g);])
end

function fixed_ktuples(g::PermGroupElem, k::Int64, L::Vector{Int64})
    fixed_pts=fixed_points(g,L)
    return prod([fixed_pts:-1:fixed_pts-k+1;])
end

function fixed_ktuples(g::PermGroupElem, k::Int64)
    return fixed_ktuples(g, k, [1:degree(g);])
end

function fixed_pts_sample(RD::RecoveryDatum, L::Vector{Int64})
    g = sample(RD)
    return fixed_points(g,L)
end

function fixed_pts_sample(RD::RecoveryDatum)
    return fixed_pts_sample(RD, [1:degree(RD);])
end

# number of fixed k-tuples by a sample of RD over L ⊆ [n]
function fixed_ktuples_sample(RD::RecoveryDatum, k::Int64, L::Vector{Int64})
    g = sample(RD)
    return fixed_ktuples(g,k,L)
end


function fixed_ktuples_sample(RD::RecoveryDatum, k::Int64)
    return fixed_ktuples_sample(RD, k, [1:degree(RD);])
end


function ktransitivity_test(RD::RecoveryDatum, k::Int; L=[1:degree(RD);], alpha=0.01, method = :hoeffding)
     @assert k<=length(L) "the provided `k` exceeds the size of the set the group is acting on"
    upper_for_low=1
    lower_for_high=2- RD.p_bound
    category = nothing
    max_rv_val=P(length(L), k)

    if method == :hoeffding
        category = HoeffdingDistinguisher(
            () -> fixed_ktuples_sample(RD, k, L), 
            upper_for_low, 
            lower_for_high, 
            alpha, 
            min_rv_val=0, 
            max_rv_val= max_rv_val, 
            boolean_valued=false, 
            delta = (1- RD.p_bound)/(2*max_rv_val))
    #elseif method == :explicit
        #category = explicit_distinguisher(() -> fixed_ktuples_sample(RD, k), upper_for_ktransitive, lower_for_nonktransitive, alpha)
    else
        error("Unknown method: $method. Use :hoeffding or :explicit")
    end 

    if category == :high 
        return false
    else
        return true
    end

    return
end

function smallest_divisor(n::Int64)
    if is_even(n)
        return 2
    else
        for d in 3:2:isqrt(n)
            if n % d == 0
                return Int(d)
            end
        end
    end

    return n
end

#include checking if n is prime?
function transitivity_test(RD::RecoveryDatum; L=[1:degree(RD);], alpha=0.01, method = :hoeffding, update = false)
    result=ktransitivity_test(RD, 1; L=L, alpha=alpha, method = method)
    #assuming we know G is non-giant
    n=degree(RD)
    if result==true
        if update == true
            add_group_property(RD, is_transitive)
            if is_prime(n) #G is primitive
                update_order_bound(RD, 4^n)
                #update_B_bound(RD, 4^n/factorial(big(n)))
                add_group_property(RD, is_primitive)
            else
                l=smallest_divisor(n) #smallest prime divisor
                
                update_order_bound(RD, BigInt((n/l)*factorial(l)*factorial(Int(n/l))^l))
                #update_B_bound(RD, Float64(RD.order_bound/factorial(big(n))))
            end
        end
        return true
    else #if G is intransitive
        if update == true
            add_group_property(RD, !is_transitive)
            update_order_bound(RD, factorial(big(n-1)))
            #update_B_bound(RD, 1/n)
        end
        return false
    end
end

################# ORBIT RECOVERY TESTS #################

#= example of intransitive G degree 20 that is not a young subgroup 
julia> G=permutation_group(20, [cperm( [1,2,7,6,3],[4,5]), cperm( [1,20,13,4,17,5,19,2,9,16,7,6,8,18,10,3,11],[12,14,15]) ])

julia> oG=map(collect, orbits(gset(G)))
2-element Vector{Vector{Int64}}:
 [1, 2, 20, 7, 9, 13, 6, 16, 4, 3, 8, 5, 17, 11, 18, 19, 10]
 [12, 14, 15]

G= permutation_group(20, [cperm([1,2,6,5],[3,4]),
 cperm([2,4]),
 cperm([1,18,16,3,10,19,15,2,13],[4,9],[5,6,20,7],[8,14,12,17])])

OG=map(collect, orbits(gset(G)))


G=permutation_group(10, [ cperm( [1,5,9,2],[3,4,7,10]), cperm([1,5,7,9,3,2,4]) ] )

 OG=map(collect, orbits(gset(G)))
3-element Vector{Vector{Int64}}:
 [1, 5, 9, 7, 2, 3, 10, 4]
 [6]
 [8]

 Oscar.describe(G)
"(C2 x C2 x C2) : PSL(3,2)"
=#

#
function respects_orbit_structure(g::PermGroupElem, O::Vector{Vector{Int64}})
    return g in young_subgroup(O)
end

# Returns a permutation property function that tests if a perm respects the orbit structure O
function orbit_permutation_property(O::Vector{Vector{Int64}})
    return (g::PermGroupElem) -> respects_orbit_structure(g, O)
end

function orbit_agreement_sample(RD::RecoveryDatum, i::Int64, j::Int64)
    sample(RD)(i)==j
end

#algorithm 8: determine if i~j (ie i,j are in the same orbit)
function orbit_agreement(RD::RecoveryDatum, i::Int64, j::Int64; alpha::Float64=0.01, method = :hoeffding)
    n=degree(RD)
    @assert i<=n && j<=n "The provided i,j are invalid"
    upper_for_low=RD.p_bound/n
    lower_for_up=1/n
    if method == :hoeffding
        category = HoeffdingDistinguisher(()->orbit_agreement_sample(RD, i, j) , upper_for_low, lower_for_up, alpha, boolean_valued=true)
    #elseif method == :explicit
        #category = explicit_distinguisher(() -> fixed_ktuples_sample(RD, k), upper_for_ktransitive, lower_for_nonktransitive, alpha)
    elseif method == :explicit
        category = explicit_distinguisher(() -> orbit_agreement_sample(RD, i, j), upper_for_low, lower_for_up, alpha)
    else
        error("Unknown method: $method. Use :hoeffding or :explicit")
    end 
    if category == :high
        return true
    else
        return false
    end

end

#algorithm 9. return the orbit of G containing i
function single_orbit_recovery(RD::RecoveryDatum, i::Int64; alpha=0.01)
    n=degree(RD)
    orbit_of_i=Vector{Int}()
    u=zeros(Int, n)
    c=(1+RD.p_bound)/(2*n)
    N=HoeffdingSampleSize(alpha, (1-RD.p_bound)/(2*n))
    for _=1:N 
        u[sample(RD)(i)]+=1
    end
    for j=1:n
        if u[j]/N>c 
            push!(orbit_of_i, j)
        end
    end
    return orbit_of_i
end

#algorithm 10
function orbit_recovery(RD::RecoveryDatum; alpha=0.01, update=false)
    unclassified=[1:degree(RD);]
    orbits = Vector{Vector{Int}}()
    while length(unclassified)>1
        i = unclassified[1]
        i_orbit = single_orbit_recovery(RD, i, alpha=alpha)
        unclassified=setdiff(unclassified, i_orbit)
        push!(orbits, i_orbit)
    end
    if !is_empty(unclassified) ##Get rid of this edge case, jsut make it run for all case in one loop
        push!(orbits, unclassified)
    end
    if update == true
        YG=young_subgroup(orbits)
        add_group_property(RD, is_subgroup_of(YG))
        add_permutation_property(RD, orbit_permutation_property(orbits))
        update_order_bound(RD, BigInt(order(YG)))
        update_B_bound(RD, Float64(BigInt(order(YG))/factorial(big(degree(RD)))))
    end
    return orbits
end






################# ORBIT COARSENING & CONFIRMATION TEST ####################

#given Δ a partition of [n], return the Young subgroup S_Δ
function young_subgroup(Δ::Vector{Vector{Int64}}; n = sum(length.(Δ)))
    gens = PermGroupElem[]
    for block in Δ
        block_size = length(block)
        if block_size >= 2
            for i in 1:block_size-1
                push!(gens, cperm([block[i], block[i+1]]))
            end
        end
    end
    return permutation_group(n, gens)
end

#=
#given a subset Δ of [1..n], return the young subgroup S_Δ x S_[n]\Δ
function young_subgroup(n::Int64, Δ::Vector{Int64})
    L=[Δ, setdiff([1:n;], Δ)]
    return young_subgroup(L)
end=#

#use the subgroup test with H = S_Δ
function orbit_coarsening_test(RD::RecoveryDatum, Δ::Vector{Vector{Int64}}; alpha=0.01, method = :hoeffding, update = false)
    return subgroup_test(RD, young_subgroup(Δ); alpha=alpha, method=method, update = update)
end

#=
#if Δ is a single orbit/vector
function orbit_coarsening_test(RD::RecoveryDatum, Δ::Vector{Int64}; alpha=0.01, method = :hoeffding)

    return subgroup_test(RD, young_subgroup(degree(RD), Δ); alpha=alpha, method=method)
end
=#




#algorithm 11: confirm Δ=[Δ1, ..., Δm] are orbits where Δ is a partition of [1..n]
function orbit_confirmation(RD::RecoveryDatum, Δ::Vector{Vector{Int64}}; alpha=0.01, method = :hoeffding, update = true)

    #contains_orbits=orbit_coarsening_test(rd, Δ; alpha=1-(sqrt(1-alpha)), method=method)

    if orbit_coarsening_test(RD, Δ; alpha=1-(sqrt(1-alpha)), method=method, update = update)==false
        verbose && println("The given partition is not a union of orbits of G")
        return false
    else
        add_permutation_property(RD, (g::PermGroupElem) -> g in young_subgroup(Δ))
        exact_orbits=Vector{Int64}[]
        flag=true
        for d in Δ
            if ktransitivity_test(RD, 1, L=d, alpha=1-(sqrt(1-alpha)), method=method)==true
                push!(exact_orbits, d)
            else
                flag=false
                verbose && println("$d is not an exact orbit of G (it is a union of orbits)")
            end
        end

        if flag == true
            verbose && println("The given partition is exactly the set of orbits of G")
            return true
        else
            verbose && println("From the given partition, only $exact_orbits are exact orbits. The rest are unions of orbits")
            return false#, exact_orbits
        end
    end

end

function confirm_orbit(G::PermGroup, S::Vector{PermGroupElem}, i::Int, j; p=0.3)
    n=degree(RD) ; N = length(S)
    count = 0
    for g in S
        if g(i)==j
            count += 1
        end 
    end

    c=(1+p)/(2*n)
    return count/N > c
end




#Algorithm 11: confirm a single orbit Δ
function single_orbit_confirmation(RD::RecoveryDatum, Δ::Vector{Int64}; alpha=0.01, method = :hoeffding, update_property = false)

    n=degree(RD)
    D=Vector{Int64}[ Δ, setdiff([1:n;], Δ ) ]

    return orbit_confirmation(RD, D, alpha=alpha, method=method, update_property=update_property)

end

function single_orbit_confirmation2(RD::RecoveryDatum, Δ::Vector{Int64}; alpha=0.01, method = :hoeffding, update_property = false)
    n=degree(RD)
    L=Vector{Int64}[ Δ , setdiff([1:n;], Δ )]

    if orbit_coarsening_test(RD, L; alpha=1-(sqrt(1-alpha)), method=method)==false
        verbose && println("$Δ is not a union of orbits of G")
        return false
    else
        add_subgroup_permutation_property(RD, young_subgroup(Δ))
  
        @time flag=ktransitivity_test(RD, 1, L=Δ, alpha=1-(sqrt(1-alpha)), method=method)
        #if ktransitivity_test(RD, 1, L=Δ, alpha=1-(sqrt(1-alpha)), method=method)==true
        if flag==true
            verbose && println("$Δ is an orbit of G ")
            #if update_property==false
             #   pop!(RD.P)
            #end
            return true      
        else
            verbose && println("$Δ is a union of orbits")
            return false#
        end

    end

end




############### SUPERGROUP TEST ###############


# find_supergroup() is a function that samples from the recovery datum and add to a list of generators for a group H. If H does not satisfy every rd.Q, add another sample into the list of generators. If H satisfies every rd.Q, then check use subgroup test to check if G is a subgroup of H. If G is not a subgroup of H, then return to adding samples to the list and checking rd.Q. If both steps are satisfied, then return H.

"""
    find_supergroup(RD::RecoveryDatum; Qbar, update=false)
Attempts to find a supergroup H of the unknown group G sampled by RD such that H satisfies every property in Qbar (a list of group properties that are known to hold for every supergroup of G, e.g., being transitive, the minimal order). If such a group H is found, it is returned.
"""
function find_supergroup(RD::RecoveryDatum; Qbar = Vector{Function}(), alpha=0.1)
    gens = Vector{PermGroupElem}()
    n = degree(RD)
    start_samples = RD.n_samples
    found = false
    while found == false
        push!(gens, sample(RD))
        H = permutation_group(n, gens)
        if all(q(H) for q in Qbar)
            if subgroup_test(RD, H, alpha=alpha, update = false) == true
                found = true
                println("\nSupergroup generated by ", length(gens), " samples.")
                return H, RD.n_samples - start_samples
            end
        else
            continue
        end
    end

    return error("Supergroup was not found")
end

function categorize_supergroups(G::PermGroup, Hs::Vector{PermGroup})
    categories = Dict("A" => 0, "B" => 0, "C" => 0, "D" => 0)
    for H in Hs
        if !is_subset(G, H)
            categories["A"] += 1  # Not a supergroup
        elseif is_transitive(H) && (is_natural_alternating_group(H) || is_natural_symmetric_group(H)) #! must be transitive, we want natural alternating of the correct degree!
            categories["B"] += 1  # Either A_n or S_n
        elseif H == G
            categories["C"] += 1  # Exactly G
        else
            categories["D"] += 1  # Proper supergroup (not A_n or S_n)
        end
    end
    return categories
end

#=
CCS = map(x->x.repr,conjugacy_classes_subgroups(symmetric_group(10)))
i=0
all_categories = []
for G in CCS
    i=i+1
    println("------------------",i)
    rd = RecoveryDatum(G, 0.2)
    Gdata = [find_supergroup(rd; alpha = 0.01) for i in 1:100]
    categories = categorize_supergroups(G, [H for (H, _) in Gdata])
    push!(all_categories, categories)
end
=#

#returns a pie chart of N many iterations of find_supergroup(rd), as well as a histogram of the number of samples used to find each supergroup
function visualize_findspg(RD, G; N=1000, Qbar=Vector{Function}(), alpha=0.1)
    n = degree(RD)
    An = alternating_group(n)
    Sn = symmetric_group(n)
    
    categories1 = Dict("A" => 0, "B" => 0, "C" => 0, "D" => 0)
    categories2 = Dict("A" => 0, "B" => 0, "C" => 0, "D" => 0)
    sample_counts1 = Vector{Int64}()
    sample_counts2 = Vector{Int64}()
    g1=Vector{PermGroup}()
    g2=Vector{PermGroup}()
    
    #first run with no group properties
    for _=1:N
        H, samples_used1 = find_supergroup(RD; Qbar=Vector{Function}(), alpha=alpha)
        push!(sample_counts1, samples_used1)
        push!(g1, H)
        # Categorize the result
        if !is_subset(G, H)
            categories1["A"] += 1  # Not a supergroup
        elseif H == An || H == Sn
            categories1["B"] += 1  # Either A_n or S_n
        elseif H == G
            categories1["C"] += 1  # Exactly G
        else
            categories1["D"] += 1  # Proper supergroup (not A_n or S_n)
        end
    end

    #second run with given Qbar
    for _=1:N
        H, samples_used2 = find_supergroup(RD; Qbar=Qbar, alpha=alpha)
        push!(sample_counts2, samples_used2)
        push!(g2, H)
        # Categorize the result
        if !is_subset(G, H)
            categories2["A"] += 1  # Not a supergroup
        elseif H == An || H == Sn
            categories2["B"] += 1  # Either A_n or S_n
        elseif H == G
            categories2["C"] += 1  # Exactly G
        else
            categories2["D"] += 1  # Proper supergroup (not A_n or S_n)
        end
    end
    
    # Create pie chart with categories
    labels = ["A: Not supergroup", "B: A_n or S_n", "C: Exactly G", "D: Other supergroup"]
    values1 = [categories1["A"], categories1["B"], categories1["C"], categories1["D"]]
    values2 = [categories2["A"], categories2["B"], categories2["C"], categories2["D"]]
    
    p1 = pie(labels, values1, title="Find Supergroup Results (N=$N)")

    p2 = pie(labels, values2, title="Find Supergroup Results with Qbar (N=$N)")

    pie_plot = plot(p1, p2, layout=(1,2), size=(1000,400))
    
    min_samples = minimum(vcat(sample_counts1, sample_counts2))
    max_samples = maximum(vcat(sample_counts1, sample_counts2))
    range_samples = max_samples - min_samples
    tick_spacing = max(1, div(range_samples, 10))  # At most 10 ticks

    bin_edges = min_samples:tick_spacing:(max_samples + tick_spacing)
    xtick_vals = bin_edges
    #=
    max_count2 = maximum([count(x -> bin_edges[i] <= x < bin_edges[i+1], sample_counts2) for i in 1:length(bin_edges)-1])
    max_count = max(max_count1, max_count2)
    ylim_upper = ceil(Int, max_count * 1.15)
    =#


    #max_count = max(maximum(values(tally(sample_counts1))), maximum(values(tally(sample_counts2))))
    #ylim_upper = ceil(Int, max_count * 1.15)  # Add 15% headroom

    # Create histogram
    hist_plot1 = histogram(sample_counts1, 
                        bins = bin_edges, 
                        #bar_width=:relative,
                        xticks = xtick_vals,
                        xlabel="Samples Used",
                        #ylims=(0, ylim_upper),
                        ylabel="Frequency",
                        title="Samples Used in find_supergroup G", 
                        label = "Without Qbar",
                        alpha=0.7,
                        color =:blue,
                        linewidth=0)
    
    hist_plot2 = histogram(sample_counts2, 
                        bins = bin_edges, 
                        #bar_width=:relative,
                        xticks = xtick_vals,
                        xlabel="Samples Used",
                        #ylims=(0, ylim_upper),
                        ylabel="Frequency",
                        title="Samples Required to Find Supergroup with Qbar", 
                        label = "With Qbar",
                        alpha=0.7,
                        color =:red,
                        linewidth=0)
    hist_plot = plot(hist_plot1, hist_plot2, layout=(1,2), size=(1000,400))
    return pie_plot, hist_plot, categories1, categories2
end

"""
function find_supergroup(RD::RecoveryDatum; Q=RD.Q, update=false)
    gens=Vector{PermGroupElem}()
    n=degree(RD)
    c=0 #counter for while loop iterations
    while true
        c+=1
        if c>=1000
            break
        end

        if is_empty(gens) #permutation group on empty set is not well defined
            push!(gens, sample(RD))
        end


        H=permutation_group(n, gens)
        if subgroup_test(RD, H, update = false)==true
            if all(q(H) for q in Q)
                print("loop iterations: ", c,"\n")
                if update == true
                    add_group_property(RD, is_subgroup_of(H))
                    add_permutation_property(RD, (g::PermGroupElem) -> g in H)
                    update_order_bound(RD, BigInt(order(H)))
                    update_B_bound(RD, Float64(BigInt(order(H))/factorial(big(n))))
                end
                return H
            else
                empty!(gens)
                print("restarting")
                continue
            end
        else
            push!(gens, sample(RD))
            print("generating set not big enough \n")
            continue
        end
        
    end
    return error("Supergroup was not found")
end
"""

############ ESTIMATING P FOR INTRANSITIVE GROUPS #########

function binomial_confidence_interval(successes::Int64, trials::Int64; alpha=0.01, method=:wilson, upper_bound = false)
    @assert 0 <= successes <= trials "successes must be between 0 and trials"
    @assert 0 < alpha < 1 "alpha must be in (0, 1)"
    
    p_hat = successes / trials
    z = quantile(Normal(), 1 - alpha/2)
    
    if method == :wilson
        # Wilson score interval (more accurate, especially for extreme p values)
        denominator = 1 + z^2/trials
        center = (p_hat + z^2/(2*trials)) / denominator
        margin = z * sqrt(p_hat*(1-p_hat)/trials + z^2/(4*trials^2)) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
    elseif method == :normal
        # Normal approximation (simpler but less accurate for extreme p values)
        se = sqrt(p_hat * (1 - p_hat) / trials)
        margin = z * se
        
        lower = max(0, p_hat - margin)
        upper = min(1, p_hat + margin)
        
    else
        error("Unknown method: $method. Use :wilson or :normal")
    end
    if upper_bound
        return upper
    else
        return (estimate=p_hat, lower=lower, upper=upper, confidence_level=1-alpha)
    end

end

    """
    Estimate the error rate p by sampling N times and checking membership in young_subgroup(O).
    Returns (p_estimate, lower_bound, upper_bound) for the confidence interval.
    """
function estimate_error_from_orbits(RD::RecoveryDatum, O::Vector{Vector{Int64}}, N::Int64; alpha=0.01, upper_bound=true)


    YG = young_subgroup(O)
    errors = count([!(RD.X() in YG) for _ in 1:N])
    
    return binomial_confidence_interval(errors, N; alpha=alpha, method=:wilson, upper_bound=upper_bound)
end

########################## TRANSITIVE CONSTITUENTS ##########################

function find_transitive_constituents(RD::RecoveryDatum, orbits::Vector{Vector{Int64}}; alpha=0.01)
    add_permutation_property(RD, (g::PermGroupElem) -> g in young_subgroup(orbits))
    constituents = Vector{PermGroup}()
    l=length(orbits)

    for i=1:l
        orbit=orbits[i]
        YG=young_subgroup([orbit, vcat(orbits[i+1:end]...)])
        YG1=young_subgroup( [orbit], n=degree(RD)) 
        proj = hom(YG, YG1, gens(YG),  replace(x -> x in gens(YG1) ? x : one(YG1), gens(YG)))
        n_orbit = length(orbit)
        new_rd = RecoveryDatum( 
            X= ()->proj(sample(RD)),
            n= degree(RD), 
            p_bound= RD.p_bound, 
            order_bound = factorial(big(n_orbit)), 
            B_bound=RD.B_bound)
        add_group_property(new_rd, is_transitive)


       # do group recovery on new_rd to find constituent
        push!(constituents, constituent)
    end
    return constituents
end

################### GROUP RECOVERY ALGORITHMS ########################

"""
    NaiveGroupRecovery(RD::RecoveryDatum, k::Int64)
Algorithm 1. Returns the permutation group generated by k random permutations sampled from the recovery datum RD.

"""
function NaiveGroupRecovery(RD::RecoveryDatum, k::Int64)
    return permutation_group(degree(RD), [sample(RD) for _ in 1:k])
end


"""
    Q_ErrorDetection(RD::RecoveryDatum, k::Int64; Q=RD.Q)
Algorithm 2. Repeatedly calls NaiveGroupRecovery(RD, k) until a group G is found that satisfies every property in Q. Returns G. 
"""
function Q_ErrorDetection(RD::RecoveryDatum, k::Int64; Q=RD.Q)
    G=NaiveGroupRecovery(RD, k)
    counter=0
    while all([q(G) for q in Q])==false
        if counter>=1000
            return error("Could not find a group satisfying Q after 1000 iterations")
        end
        counter+=1
        G=NaiveGroupRecovery(RD, k)
    end
    return G
end

"""
    NiAGRA(RD::RecoveryDatum, k::Int64; alpha=0.01, delta=0.25)
Algorithm 3. Performs N=HoeffdingSampleSize(alpha, delta) iterations of Q_ErrorDetection(RD, k) and returns the mode of the resulting groups.
"""
function NiAGRA(RD::RecoveryDatum,k::Int64; alpha=0.01, delta=0.25, Q_error=true) 
    #default is N=37, but when success rate is known, it is reduced according to delta=success_rate - 0.5
    N=HoeffdingSampleSize(alpha,delta)
    println("")
    counts=Dict{PermGroup, Int64}()
    for i=1:N
        if Q_error
            G = Q_ErrorDetection(RD, k)
        else
            G = NaiveGroupRecovery(RD, k)
        end
        counts[G]=get(counts, G, 0) + 1
    end
    first_mode = argmax(counts)
    println("Answer seen ", 100*counts[first_mode]/N, "% of the time")
    return(first_mode)

end

###################### MAIN ALGORITHM ################

function main_group_recovery(RD::RecoveryDatum; alpha=0.01, method = :hoeffding)
    is_giant = giant_test(RD; alpha=alpha, method=method, update=true)
    is_transitive = nothing
    orbits = nothing
    if is_giant == true
        verbose && println("G is a giant. Checking if alternating or symmetric:")

        is_alternating = alternating_test(RD; alpha=alpha, method=method)
        if is_alternating == true
            verbose && println("G is the alternating group A_n")
            return alternating_group(degree(RD))
        else
            verbose && println("G is the symmetric group S_n")
            return symmetric_group(degree(RD))
        end
        return

    else #if G is non-giant, check transitivity
        verbose && println("G is a non-giant. Checking transitivity:")
        is_transitive = transitivity_test(RD; alpha=alpha, method=method, update=true)


        if is_transitive == true
             

            verbose && println("G is a transitive non-giant.")
            verbose && println("Checking success rate:")

            success_rate=success_rate_check(RD, Mtilde(RD)+4)

            if success_rate<0.5
                println("\nSuccess rate too low, must find more properties or improve bounds")
                println("Finding a supergroup of G...")
                for _=1:100
                    H=find_supergroup(RD; update=true)
                    success_rate=success_rate_check(RD, Mtilde(RD)+4)
                    if success_rate>0.5
                        break
                    end
                end  
            end
            
            verbose && println("\nEstimated number of generators needed: $(Mtilde(RD)+4), performing NiAGRA with base success rate $success_rate.")
               
            G=NiAGRA(RD, Mtilde(RD)+4; alpha=alpha, delta=success_rate-0.5)
            
            return G
            
        else #if G is intransitive
            verbose && println("G is intransitive. Checking success rate:")
            success_rate=success_rate_check(RD, Mtilde(RD)+4)
            if success_rate < 0.5
                println("\nSuccess rate too low, must find more properties or improve bounds")
                println("\nFinding the orbits of G...")
                orbits=orbit_recovery(RD; alpha=alpha, update=true) 
                success_rate=success_rate_check(RD, Mtilde(RD)+4)
            end
            if success_rate < 0.5
                println("\nSuccess rate still too low, must find more properties or improve bounds on error")
                println("\nEstimating error bounds from orbits...")
                pu = estimate_error_from_orbits(RD, orbits, 1000; alpha=0.01, upper_bound=true)
                update_p_bound(RD, pu)
                success_rate=success_rate_check(RD, Mtilde(RD)+4)
            end
            if success_rate < 0.5
                println("Finding a supergroup of G...")
                for _=1:100
                    H=find_supergroup(RD; update=true)
                    success_rate=success_rate_check(RD, Mtilde(RD)+4)
                    if success_rate>0.5
                        break
                    end
                end  
            end

            if success_rate<0.5
                println("\nSuccess rate still too low, must find more properties or improve bounds")
                println("Cannot proceed with group recovery.")
                return
            else
                verbose && println("\nEstimated number of generators needed: $(Mtilde(RD)+4), performing NiAGRA with base success rate $success_rate.")
                G=NiAGRA(RD, Mtilde(RD)+4; alpha=alpha, delta=success_rate-0.5)
                return G
            end
        end
    end
end



###################### HELPER FUNCTIONS ##########################



function weylgroup()
    return permutation_group(27,[cperm([1, 10, 13],[2, 24, 6],[3, 17, 11],[4, 23, 8],[5, 26, 25],[7, 18, 12],[9, 20, 16],[14, 27, 19],[15, 21, 22]), cperm([1, 18, 13, 22, 10, 11],[2, 4, 21, 27, 9, 15],[3, 20, 26, 5, 14, 7],[6, 25, 23],[8, 12, 17, 24, 16, 19])  ] )
end

function worst_group(n)
    P=[cperm([[i, i+1] for i=1:2:(2^n-1) ])]
    m=Int(2^(n-1))
    push!(P,cperm([[i, i+m] for i=1:m ]))
    while m>2
        T=Vector{Vector{Int64}}()
        for i=0:div(2^n,m)-1
            L=[1:m;] .+i*m
            [push!(T, [L[1+j], L[end-j]]) for j=0:div(m,2)-1]
        end
        push!(P,cperm(T))
        m=divexact(m,2)
    end
    return permutation_group(2^n, P) 
end


function tally(S)
    D=Dict{typeof(S[1]),Int}()
    for s in S
        D[s]=get(D,s,0)+1
    end
    return D
end




######################  BLOCK RECOVERY & PRIMITIVITY TEST ##########################

#=
# Minimal Block Recovery function. Given RD::RecoveryDatum and N a number of samples, let X be a sample from RD. Let X' be the image of X under the inclusion Sn -> S_(n choose 2) via action on distinct paris of [n]. Determine the orbits of G^(2) by applying orbit recovery on RD' that samples from X'. From the orbits of G^(2), determine the minimal blocks of G.
#example:  G=transitive_group(10, 40), blocks are [1,3,5,7,9] and [2,4,6,8,10]
function minimal_block_recovery(RD::RecoveryDatum; N::Int64=1000, alpha::Float64=0.01)
    n = degree(RD)
    Omega=gset( symmetric_group(n), [[1,2]])
    acthom=action_homomorphism( gset( Omega))
    #new RD' that is the image of RD under action on distinct pairs
    RD_pairs= RecoveryDatum( X=()->acthom(sample(RD)), n=degree(RD)*(degree(RD)-1), p_bound=RD.p_bound, order_bound=factorial(big(degree(RD)*(degree(RD)-1))), B_bound=RD.B_bound)

    # Define the new sampling function for RD'
    orbits_pairs = orbit_recovery(RD_pairs; alpha=alpha)
    # From the orbits of G^(2), determine the minimal blocks of G
    blocks = Vector{Vector{Int64}}()
    for orbit in orbits_pairs
        edges=Vector{Vector{Int64}}()
        for i in orbit
            push!(edges, Omega[i])
        end

    end
    for orbit in orbits_pairs
        # Each orbit is a set of pairs (i,j)
        pair_set = Set{Int64}()
        for pair in orbit
            i, j = pair
            push!(pair_set, i)
            push!(pair_set, j)
        end
        block = collect(pair_set)
        push!(blocks, block)
    end
    return blocks
end

=#
######################## GROUP PROPERTIES ##########################









#=
function orbit_coarsening_sample(RD::RecoveryDatum, n_partition::Vector{Int})
    g = sample(RD)
    # Create the product of symmetric groups based on the partition
    H = wreath_product_groups(n_partition)
    return g in H
end

function wreath_product_groups(n_partition::Vector{Int})
    # TODO: Implement creation of product of symmetric groups
    # This should create ∏ S_{n_i} for n_i in n_partition
    # For now, return identity as placeholder
    n_total = sum(n_partition)
    return symmetric_group(n_total)  # Placeholder
end

function orbit_coarsening_upper_if_low(n_partition::Vector{Int}, n, p)
    # TODO: Implement upper bound for proportion in orbit coarsening group when G is in :low category
    return 0.0  # Placeholder
end

function orbit_coarsening_upper_if_low(RD::RecoveryDatum, n_partition::Vector{Int})
    n = degree(RD)
    p = RD.p_bound
    orbit_coarsening_upper_if_low(n_partition, n, p)
end

function orbit_coarsening_lower_if_high(n_partition::Vector{Int}, n, p)
    # TODO: Implement lower bound for proportion in orbit coarsening group when G is in :high category
    return 1.0  # Placeholder
end

function orbit_coarsening_lower_if_high(RD::RecoveryDatum, n_partition::Vector{Int})
    n = degree(RD)
    p = RD.p_bound
    orbit_coarsening_lower_if_high(n_partition, n, p)
end

function orbit_coarsening_test(RD::RecoveryDatum, n_partition::Vector{Int}; alpha=0.01, method = :hoeffding)
    upper_for_low = orbit_coarsening_upper_if_low(RD, n_partition)
    lower_for_high = orbit_coarsening_lower_if_high(RD, n_partition)
    category = nothing
    if method == :hoeffding
        category = HoeffdingDistinguisher(() -> orbit_coarsening_sample(RD, n_partition), upper_for_low, lower_for_high, alpha)
    else
        category = explicit_distinguisher(() -> orbit_coarsening_sample(RD, n_partition), upper_for_low, lower_for_high, alpha)
    end
    if category == :high
        return true
    else
        return false
    end
end
=#

end # module GroupRecovery
