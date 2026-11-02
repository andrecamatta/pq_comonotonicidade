# Comonotonicidade.jl
#
# Módulo numérico central da Parte 4 da série: agregação de risco entre
# segmentos (setores de crédito, grupos de ações) sob diferentes hipóteses de
# dependência, e o pior caso de VaR via rearrangement algorithm.
#
# Referências: DHAENE et al. (2002a, 2002b); EMBRECHTS, PUCCETTI, RÜSCHENDORF
# (2013); PUCCETTI, RÜSCHENDORF (2012); BERNARD, RÜSCHENDORF, VANDUFFEL (2017).

module Comonotonicidade

using Statistics
using StatsBase
using Random
using Distributions
using LinearAlgebra

export kendall_tau_matrix, mean_kendall_tau, comonotonicity_coefficient,
       comonotonic_sum_sample, independent_sum_sample, gaussian_copula_sum_sample,
       rearrange_worst_var, es_var, block_bootstrap_ci

# -----------------------------------------------------------------------
# Medidas de dependência
# -----------------------------------------------------------------------

"Matriz de Kendall's tau par a par entre colunas de X (observações nas linhas, segmentos nas colunas)."
function kendall_tau_matrix(X::AbstractMatrix{<:Real})
    n, d = size(X)
    T = Matrix{Float64}(I, d, d)
    for i in 1:d, j in (i+1):d
        t = corkendall(X[:, i], X[:, j])
        T[i, j] = t
        T[j, i] = t
    end
    return T
end

"Média das entradas fora da diagonal da matriz de Kendall's tau (dependência média par a par)."
function mean_kendall_tau(X::AbstractMatrix{<:Real})
    T = kendall_tau_matrix(X)
    d = size(T, 1)
    d < 2 && return NaN
    s = 0.0
    n = 0
    for i in 1:d, j in (i+1):d
        s += T[i, j]
        n += 1
    end
    return s / n
end

"""
    comonotonicity_coefficient(X)

Coeficiente de comonotonicidade κ(X) ∈ [0,1] (DHAENE; VANDUFFEL; GOOVAERTS):
κ = (Var(S) - ΣVar(Xᵢ)) / ((Σσᵢ)² - Σσᵢ²), com S = ΣXᵢ.
κ=0 corresponde a independência (par a par descorrelacionada), κ=1 a
comonotonicidade perfeita (correlação par a par igual a 1 em todos os pares).
"""
function comonotonicity_coefficient(X::AbstractMatrix{<:Real})
    n, d = size(X)
    S = sum(X, dims=2)[:, 1]
    var_S = var(S)
    sigmas = [std(X[:, i]) for i in 1:d]
    soma_var = sum(sigmas .^ 2)
    soma_sigma_sq = sum(sigmas)^2
    denom = soma_sigma_sq - soma_var
    denom <= 0 && return NaN
    return (var_S - soma_var) / denom
end

# -----------------------------------------------------------------------
# Simulação da soma sob diferentes hipóteses de dependência
# -----------------------------------------------------------------------

"Amostra Monte Carlo da soma sob independência entre marginais (vetor de distribuições univariadas)."
function independent_sum_sample(marginais::Vector{<:UnivariateDistribution}, nsim::Int; rng=Random.default_rng())
    d = length(marginais)
    S = zeros(nsim)
    for i in 1:d
        S .+= rand(rng, marginais[i], nsim)
    end
    return S
end

"""
    comonotonic_sum_sample(marginais, nsim)

Amostra exata da soma comonotônica: um único U~Unif(0,1) é aplicado à
inversa de cada marginal, S^c(U) = Σ Fᵢ⁻¹(U). Dá VaR/ES por fórmula fechada
(aditividade de quantis), aqui obtida por simulação da variável U comum.
"""
function comonotonic_sum_sample(marginais::Vector{<:UnivariateDistribution}, nsim::Int; rng=Random.default_rng())
    d = length(marginais)
    U = rand(rng, nsim)
    S = zeros(nsim)
    for i in 1:d
        S .+= quantile.(marginais[i], U)
    end
    return S
end

"""
    gaussian_copula_sum_sample(marginais, correl, nsim)

Amostra da soma sob uma cópula Gaussiana com matriz de correlação `correl`
(calibrada empiricamente, ex. via Kendall's tau convertido por ρ=sin(πτ/2)),
aplicada às marginais `marginais` por transformação integral de probabilidade.
Aproximação padrão para "dependência calibrada", não o pior caso teórico.
"""
function gaussian_copula_sum_sample(marginais::Vector{<:UnivariateDistribution}, correl::AbstractMatrix{<:Real},
                                     nsim::Int; rng=Random.default_rng())
    d = length(marginais)
    # Garantir que a matriz é positiva semidefinida (pequena correção se necessário)
    Sigma = Symmetric((correl + correl') / 2)
    mvn = MvNormal(zeros(d), Sigma)
    Z = rand(rng, mvn, nsim)  # d x nsim
    S = zeros(nsim)
    for i in 1:d
        u = cdf.(Normal(), Z[i, :])
        S .+= quantile.(marginais[i], clamp.(u, 1e-10, 1 - 1e-10))
    end
    return S
end

"""
    rearrange_worst_var(marginais, alpha, N; maxiter=200)

Aproxima o pior caso (maior) de VaR_α(S) sob só as marginais conhecidas, via
rearrangement algorithm (PUCCETTI; RÜSCHENDORF, 2012; EMBRECHTS; PUCCETTI;
RÜSCHENDORF, 2013). Discretiza a região de cauda (α, 1) de cada marginal em N
pontos, e rearranja iterativamente cada coluna da matriz N×d para minimizar a
variância da soma das linhas (tornando-as o mais homogêneas possível, no
espírito de "complete mixability"). O pior VaR estimado é o MÍNIMO das somas
de linha da matriz rearranjada ao final.

Sem prova formal de convergência (heurística numérica, não garantida; ver
nota do próprio algoritmo em EMBRECHTS; PUCCETTI; RÜSCHENDORF). Retorna
(pior_var, matriz_rearranjada).
"""
function rearrange_worst_var(marginais::Vector{<:UnivariateDistribution}, alpha::Real, N::Int; maxiter::Int=200)
    d = length(marginais)
    X = Matrix{Float64}(undef, N, d)
    for i in 1:d
        p = [alpha + (k - 0.5) / N * (1 - alpha) for k in 1:N]
        X[:, i] = quantile.(marginais[i], p)
    end
    for it in 1:maxiter
        changed = false
        for j in 1:d
            others = sum(X[:, setdiff(1:d, j)], dims=2)[:, 1]
            order_desc = sortperm(others, rev=true)
            col_sorted_asc = sort(X[:, j])
            newcol = similar(X[:, j])
            newcol[order_desc] .= col_sorted_asc
            if !isapprox(newcol, X[:, j]; atol=1e-12)
                changed = true
            end
            X[:, j] = newcol
        end
        changed || break
    end
    rowsums = sum(X, dims=2)[:, 1]
    return minimum(rowsums), X
end

# -----------------------------------------------------------------------
# Medidas de risco
# -----------------------------------------------------------------------

"VaR (quantil) e ES (média da cauda acima do VaR) de uma amostra, no nível α."
function es_var(S::AbstractVector{<:Real}, alpha::Real)
    v = quantile(S, alpha)
    tail = S[S .>= v]
    es = isempty(tail) ? v : mean(tail)
    return (var=v, es=es)
end

# -----------------------------------------------------------------------
# Bootstrap em bloco (temporal)
# -----------------------------------------------------------------------

"""
    block_bootstrap_ci(X, statfun; nboot=2000, block_len=4, alpha=0.10, rng)

Bootstrap em bloco circular sobre as LINHAS de X (observações no tempo),
preservando a autocorrelação serial dentro de cada bloco. `statfun` recebe
uma matriz reamostrada e devolve um escalar. Retorna (estimativa_pontual,
ic_inferior, ic_superior, amostras_bootstrap).
"""
function block_bootstrap_ci(X::AbstractMatrix{<:Real}, statfun::Function;
                             nboot::Int=2000, block_len::Int=4, alpha::Real=0.10,
                             rng=Random.default_rng())
    n = size(X, 1)
    ponto = statfun(X)
    block_len = min(block_len, n)
    nblocos = cld(n, block_len)
    boots = Vector{Float64}(undef, nboot)
    for b in 1:nboot
        idx = Int[]
        for _ in 1:nblocos
            start = rand(rng, 1:n)
            for k in 0:(block_len - 1)
                push!(idx, mod1(start + k, n))
            end
        end
        idx = idx[1:n]
        boots[b] = statfun(X[idx, :])
    end
    lo = quantile(boots, alpha / 2)
    hi = quantile(boots, 1 - alpha / 2)
    return (ponto=ponto, ic_lo=lo, ic_hi=hi, amostras=boots)
end

end # module
