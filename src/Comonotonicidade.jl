# Comonotonicidade.jl
#
# Módulo numérico central da Parte 4 da série: agregação de risco entre
# segmentos (setores de crédito, grupos de ações) sob diferentes hipóteses de
# dependência, e o pior caso de VaR via rearrangement algorithm.
#
# Referências: DHAENE et al. (2002); DHAENE, LINDERS, SCHOUTENS, VYNCKE
# (2014); EMBRECHTS, PUCCETTI, RÜSCHENDORF (2013); PUCCETTI, RÜSCHENDORF
# (2012).

module Comonotonicidade

using Statistics
using StatsBase
using Random
using Distributions
using LinearAlgebra

export kendall_tau_matrix, mean_kendall_tau, comonotonicity_coefficient,
       comonotonic_rearrangement, comonotonic_sum_sample, independent_sum_sample,
       gaussian_copula_sum_sample, rearrange_worst_var, es_var, block_bootstrap_ci,
       first_differences, empirical_tail_dependence, mean_tail_dependence,
       gaussian_copula_tail_dependence

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
    comonotonic_rearrangement(X)

Rearranjo comonotônico empírico de X: cada coluna é ordenada de forma
ascendente, independentemente das demais, e as colunas ordenadas são
realinhadas pela mesma posição de linha. Essa é a construção empírica padrão
da cópula comonotônica a partir de observações (cada marginal aplicada ao
mesmo "ranking" U), usada para obter Var(Sᶜ) sem assumir correlação linear
perfeita entre pares.
"""
function comonotonic_rearrangement(X::AbstractMatrix{<:Real})
    n, d = size(X)
    Xc = similar(X)
    for i in 1:d
        Xc[:, i] = sort(X[:, i])
    end
    return Xc
end

"""
    comonotonicity_coefficient(X)

Medida de dependência ρ_c(X) de DHAENE, LINDERS, SCHOUTENS e VYNCKE (2014):
ρ_c = (Var(S) - ΣVar(Xᵢ)) / (Var(Sᶜ) - ΣVar(Xᵢ)), com S = ΣXᵢ e Sᶜ a soma
comonotônica das MESMAS observações empíricas, construída ordenando cada
coluna de forma ascendente e somando linha a linha (`comonotonic_rearrangement`).
Diferente de aproximar Var(Sᶜ) por (Σσᵢ)², que só é exato quando a correlação
par a par sob comonotonicidade é exatamente 1 (caso de marginais ligadas por
transformação linear), Var(Sᶜ) aqui é calculado diretamente do rearranjo, válido
para marginais heterogêneas e não linearmente relacionadas.

ρ_c=1 corresponde à soma já estar no rearranjo comonotônico (dependência
máxima compatível com as marginais observadas). ρ_c=0 corresponde a
covariância par a par nula em média, NÃO a independência (covariância nula
não implica independência fora do caso Gaussiano). ρ_c pode ser negativo,
quando a estrutura de dependência observada é, em média, negativamente
correlacionada.
"""
function comonotonicity_coefficient(X::AbstractMatrix{<:Real})
    n, d = size(X)
    S = sum(X, dims=2)[:, 1]
    var_S = var(S)
    sigmas = [std(X[:, i]) for i in 1:d]
    soma_var = sum(sigmas .^ 2)
    Xc = comonotonic_rearrangement(X)
    Sc = sum(Xc, dims=2)[:, 1]
    var_Sc = var(Sc)
    denom = var_Sc - soma_var
    denom <= 0 && return NaN
    return (var_S - soma_var) / denom
end

# -----------------------------------------------------------------------
# Transformação para estacionariedade e dependência de cauda
# -----------------------------------------------------------------------

"Primeiras diferenças por coluna (Δxₜ = xₜ - xₜ₋₁), uma linha a menos que X."
function first_differences(X::AbstractMatrix{<:Real})
    n, d = size(X)
    n < 2 && return X[1:0, :]
    return X[2:end, :] .- X[1:end-1, :]
end

"""
    empirical_tail_dependence(x, y; q=0.9)

Coeficientes empíricos de dependência de cauda superior (λ_U) e inferior
(λ_L) entre duas séries, por contagem de postos (NELSEN; SCHMIDT), no
limiar de quantil `q` (cauda superior) e `1-q` (cauda inferior).
λ_U = P(postoY > nq | postoX > nq), estimado por contagem direta.
"""
function empirical_tail_dependence(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; q::Real=0.9)
    n = length(x)
    rx = StatsBase.ordinalrank(x)
    ry = StatsBase.ordinalrank(y)
    thresh_hi = q * n
    thresh_lo = (1 - q) * n
    nx_hi = count(>(thresh_hi), rx)
    nx_lo = count(<=(thresh_lo), rx)
    lambda_u = nx_hi == 0 ? NaN : count(i -> rx[i] > thresh_hi && ry[i] > thresh_hi, 1:n) / nx_hi
    lambda_l = nx_lo == 0 ? NaN : count(i -> rx[i] <= thresh_lo && ry[i] <= thresh_lo, 1:n) / nx_lo
    return (lambda_u=lambda_u, lambda_l=lambda_l)
end

"Média par a par de λ_U e λ_L entre todas as colunas de X (observações nas linhas)."
function mean_tail_dependence(X::AbstractMatrix{<:Real}; q::Real=0.9)
    n, d = size(X)
    us = Float64[]
    ls = Float64[]
    for i in 1:d, j in (i+1):d
        r = empirical_tail_dependence(X[:, i], X[:, j]; q=q)
        isnan(r.lambda_u) || push!(us, r.lambda_u)
        isnan(r.lambda_l) || push!(ls, r.lambda_l)
    end
    return (lambda_u=mean(us), lambda_l=mean(ls), n_pairs=length(us))
end

"""
    gaussian_copula_tail_dependence(tau, q, nsim; rng)

Dependência de cauda λ_U, λ_L de uma cópula Gaussiana bivariada com
correlação ρ=sin(πτ/2) (mesma conversão usada para calibrar a cópula de
dependência da Parte 4), estimada por simulação com o mesmo estimador
empírico usado nos dados reais (`empirical_tail_dependence`), para
comparação direta: o quanto da dependência de cauda observada já é
explicada por uma cópula Gaussiana calibrada só no Kendall's τ médio.
"""
function gaussian_copula_tail_dependence(tau::Real, q::Real, nsim::Int; rng=Random.default_rng())
    rho = sin(pi * tau / 2)
    Sigma = [1.0 rho; rho 1.0]
    mvn = MvNormal(zeros(2), Sigma)
    Z = rand(rng, mvn, nsim)
    u1 = cdf.(Normal(), Z[1, :])
    u2 = cdf.(Normal(), Z[2, :])
    return empirical_tail_dependence(u1, u2; q=q)
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
