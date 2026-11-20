# aggregate_capital.jl
#
# Tradução das medidas de dependência em capital (ES/VaR a 99%) sobre
# distribuições CALIBRADAS: marginais ajustadas no NÍVEL da série completa de
# cada trilha (crédito, ações), combinadas com a estrutura de dependência
# estimada em cada janela a partir de dados ESTACIONÁRIOS (primeiras
# diferenças no crédito, retorno diário já estacionário nas ações), sob
# quatro hipóteses: independência, dependência calibrada, comonotonicidade
# (fórmula fechada via simulação com U comum) e pior caso de VaR via
# rearrangement algorithm dado só as marginais.
#
# Teoria de referência (ARTZNER et al., 1999; EMBRECHTS; PUCCETTI;
# RÜSCHENDORF, 2013): VaR_α(S) ≤ ES_α(S) ≤ Σᵢ ES_α(Xᵢ) para qualquer soma S
# de marginais Xᵢ conhecidas. O pior VaR via rearrangement algorithm aproxima
# o teto dessa cadeia, Σᵢ ES_α(Xᵢ) = ES_α(Sᶜ) (a soma comonotônica), não um
# valor além dele.
#
# Limitações declaradas: (1) a "unidade de capital" da trilha de crédito é a
# soma NÃO PONDERADA das taxas de inadimplência setoriais, não uma soma de
# saldo em reais nem uma métrica ponderada por exposição. É uma grandeza
# relativa para comparar os quatro cenários entre si, não uma estimativa de
# capital regulatório. (2) A marginal Gama de cada setor de crédito é
# ajustada sobre o NÍVEL da taxa de inadimplência (54 trimestres, 2012-2025),
# uma série com tendência e quebras de regime, não estacionária; a marginal
# descreve a dispersão agregada da série, não uma distribuição estacionária
# "verdadeira" de um processo gerador único. (3) A tradução em capital por
# janela só é calculada quando a janela tem observações suficientes para
# calibrar uma matriz de correlação de d dimensões (d=19 para crédito, d=10
# para ações); para o crédito, isso exclui as duas janelas de estresse
# (2015-2016, n=8; 2020, n=4), calculadas só via medidas de dependência
# (dependence_credit.jl), não traduzidas em capital aqui.

using Printf, Random, Statistics, LinearAlgebra, Distributions
include("src/Comonotonicidade.jl")
using .Comonotonicidade

const OUT = joinpath(@__DIR__, "data", "output")
const NSIM = 300_000
const ALPHA = 0.99
const TDIST_NU = 5.0  # graus de liberdade da t de Student usada para a perda de retorno (cauda mais pesada que a Normal)

"Split de linha CSV que respeita campos entre aspas duplas (podem conter vírgulas)."
function split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    inquotes = false
    for c in line
        if c == '"'
            inquotes = !inquotes
        elseif c == ',' && !inquotes
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

function load_credit()
    lines = readlines(joinpath(OUT, "scr_painel_setorial.csv"))
    rows = NamedTuple[]
    for l in lines[2:end]
        cols = split_csv_line(l)
        push!(rows, (trimestre = cols[1], cnae = cols[3], taxa = parse(Float64, cols[7])))
    end
    cnt = Dict{String,Int}()
    for r in rows; cnt[r.cnae] = get(cnt, r.cnae, 0) + 1; end
    tcount = length(unique(r.trimestre for r in rows))
    setores = sort([s for (s, c) in cnt if c == tcount])
    trimestres = sort(unique(r.trimestre for r in rows); by = q -> parse(Int, q[1:4]) * 10 + parse(Int, q[6]))
    idx = Dict(t => i for (i, t) in enumerate(trimestres))
    jdx = Dict(s => j for (j, s) in enumerate(setores))
    X = fill(NaN, length(trimestres), length(setores))
    for r in rows
        haskey(jdx, r.cnae) || continue
        X[idx[r.trimestre], jdx[r.cnae]] = r.taxa
    end
    return X, setores, trimestres
end

function load_b3()
    lines = readlines(joinpath(OUT, "b3_painel_setorial.csv"))
    rows = NamedTuple[]
    for l in lines[2:end]
        cols = split(l, ',')
        push!(rows, (data = parse(Int, cols[1]), setor = replace(cols[2], "\"" => ""), ret = parse(Float64, cols[3])))
    end
    datas = sort(unique(r.data for r in rows))
    setores = sort(unique(r.setor for r in rows))
    idx = Dict(d => i for (i, d) in enumerate(datas))
    jdx = Dict(s => j for (j, s) in enumerate(setores))
    X = fill(NaN, length(datas), length(setores))
    for r in rows
        X[idx[r.data], jdx[r.setor]] = r.ret
    end
    return X, setores, datas
end

"""
Projeta uma matriz simétrica para a matriz de correlação positiva semidefinida
mais próxima: clip de autovalores negativos, seguido de normalização
D^(-1/2) M D^(-1/2) para restaurar diagonal = 1. Sem a normalização, o clip de
autovalores por si só desloca a diagonal para fora de 1 (ex. 1,01 a 1,10 numa
matriz 19×19 quase indefinida), o que infla artificialmente a variância das
marginais quando a matriz é usada como correlação de uma cópula Gaussiana.
"""
function nearest_psd(M::AbstractMatrix{<:Real})
    S = Symmetric((M + M') / 2)
    F = eigen(S)
    vals = max.(F.values, 1e-8)
    Mpsd = Symmetric(F.vectors * Diagonal(vals) * F.vectors')
    d = sqrt.(diag(Mpsd))
    Dinv = Diagonal(1.0 ./ d)
    return Symmetric(Dinv * Mpsd * Dinv)
end

"""
Projeção de Higham (2002) para a matriz de correlação mais próxima (norma de
Frobenius), por projeções alternadas com a correção de Dykstra entre o cone
semidefinido positivo (piso `delta` nos autovalores, para a matriz ficar
estritamente positiva definida e admitir Cholesky na simulação) e o conjunto
de matrizes simétricas de diagonal 1.
"""
function nearest_corr_higham(A::AbstractMatrix{<:Real}; tol::Float64=1e-12, maxiter::Int=2000, delta::Float64=1e-8)
    d = size(A, 1)
    Y = Matrix{Float64}((A + A') / 2)
    dS = zeros(d, d)
    X = copy(Y)
    for _ in 1:maxiter
        R = Y - dS
        F = eigen(Symmetric(R))
        Xn = F.vectors * Diagonal(max.(F.values, delta)) * F.vectors'
        Xn = (Xn + Xn') / 2
        dS = Xn - R
        Yn = copy(Xn)
        for i in 1:d
            Yn[i, i] = 1.0
        end
        conv = norm(Yn - Y, Inf) / norm(Yn, Inf)
        Y = Yn
        X = Xn
        conv < tol && break
    end
    Dinv = Diagonal(1.0 ./ sqrt.(diag(X)))
    return Matrix(Symmetric(Dinv * X * Dinv))
end

"Matriz sin(πτ/2) com diagonal 1, SEM correção de positividade."
function tau_to_raw_corr(T::AbstractMatrix{<:Real})
    d = size(T, 1)
    return [i == j ? 1.0 : sin(pi * T[i, j] / 2) for i in 1:d, j in 1:d]
end

"Equicorrelação calibrada no τ médio: ρ = sin(π τ̄/2) em todos os pares (positiva definida por construção se ρ > -1/(d-1))."
function equicorr_from_tau(T::AbstractMatrix{<:Real})
    d = size(T, 1)
    tbar = sum(T[i, j] for i in 1:d for j in 1:d if i != j) / (d * (d - 1))
    rho = sin(pi * tbar / 2)
    return [i == j ? 1.0 : rho for i in 1:d, j in 1:d], tbar, rho
end

"""
Converte matriz de Kendall's tau em matriz de correlação de cópula Gaussiana (ρ=sin(πτ/2)),
corrigida para positiva definida. `metodo`: :higham (padrão, correlação mais próxima de
Higham) ou :clip (corte de autovalores com renormalização, o método usado até a versão
anterior deste código).
"""
function tau_to_gaussian_corr(T::AbstractMatrix{<:Real}; metodo::Symbol=:higham)
    R = tau_to_raw_corr(T)
    metodo == :clip && return Matrix(nearest_psd(R))
    metodo == :higham && return nearest_corr_higham(R)
    error("metodo desconhecido: $metodo")
end

"Marginal de 'perda' por setor de crédito = taxa de inadimplência trimestral (NÍVEL), ajustada a uma Gama por método dos momentos na série completa."
function fit_marginals_credit(X::AbstractMatrix{<:Real})
    d = size(X, 2)
    marg = Vector{Gamma}(undef, d)
    for i in 1:d
        col = filter(!isnan, X[:, i])
        m, v = mean(col), var(col)
        theta = v / m
        k = m / theta
        marg[i] = Gamma(k, theta)
    end
    return marg
end

"Marginal de 'perda' por setor de ações = retorno diário negado, ajustada a uma t de Student de localização-escala (ν=5, cauda mais pesada que a Normal) por método dos momentos na série completa."
function fit_marginals_b3(X::AbstractMatrix{<:Real})
    d = size(X, 2)
    marg = Vector{Any}(undef, d)
    for i in 1:d
        perdas = filter(!isnan, -X[:, i])
        m, s = mean(perdas), std(perdas)
        scale = s * sqrt((TDIST_NU - 2) / TDIST_NU)
        marg[i] = m + scale * TDist(TDIST_NU)
    end
    return identity.(marg)
end

function window_select(trimestres::Vector{String}, ini::String, fim::String)
    key(q) = parse(Int, q[1:4]) * 10 + parse(Int, q[6])
    return [key(t) >= key(ini) && key(t) <= key(fim) for t in trimestres]
end

"""
    run_track(...)

`X_marg` fornece os dados para ajustar as marginais (nível). `X_corr` e
`periodos_corr` fornecem os dados ESTACIONÁRIOS (diferenças, no crédito; o
próprio retorno, nas ações) para calibrar a matriz de correlação por janela.
`min_n_capital` é o número mínimo de observações estacionárias exigido para
calcular a tradução em capital de uma janela (abaixo disso, a janela aparece
só nas medidas de dependência, não aqui).
"""
function run_track(nome_trilha::AbstractString, X_marg::Matrix{Float64}, X_corr::Matrix{Float64},
                    periodos_corr::Vector, janelas::Vector{<:Tuple}, fit_fun::Function, ra_N::Int;
                    min_n_capital::Int=8, psd::Symbol=:higham, write_csv::Bool=true)
    Random.seed!(20261102)
    marg = fit_fun(X_marg)
    d = length(marg)
    @printf("\n=== %s (d=%d segmentos) ===\n", nome_trilha, d)

    Sindep = independent_sum_sample(marg, NSIM)
    Scomon = comonotonic_sum_sample(marg, NSIM)
    es_indep = es_var(Sindep, ALPHA)
    es_comon = es_var(Scomon, ALPHA)
    wv, _ = rearrange_worst_var(marg, ALPHA, ra_N)

    @printf("Independência:         VaR99=%.4f  ES99=%.4f\n", es_indep.var, es_indep.es)
    @printf("Comonotonicidade:      VaR99=%.4f  ES99=%.4f (= Σ ES99 marginal, teto teórico)\n", es_comon.var, es_comon.es)
    @printf("Pior caso de VaR (RA, N=%d): %.4f\n", ra_N, wv)
    @printf("  razão RA / VaR comonotônico = %.4f\n", wv / es_comon.var)
    @printf("  razão RA / ES comonotônico (teto teórico Σ ES99) = %.4f  (teoria: VaR(S) ≤ ES(S) ≤ Σ ES99, logo ≤ 1)\n",
            wv / es_comon.es)

    resultados = NamedTuple[]
    push!(resultados, (janela="independencia", n=NaN, var99=es_indep.var, es99=es_indep.es))
    push!(resultados, (janela="comonotonicidade", n=NaN, var99=es_comon.var, es99=es_comon.es))
    push!(resultados, (janela="pior_caso_RA", n=NaN, var99=wv, es99=NaN))

    for (nome, ini, fim) in janelas
        sel = if periodos_corr isa Vector{String}
            window_select(periodos_corr, ini, fim)
        else
            [p >= ini && p <= fim for p in periodos_corr]
        end
        Xw_full = X_corr[sel, :]
        completos = [all(!isnan, Xw_full[i, :]) for i in 1:size(Xw_full, 1)]
        Xw = Xw_full[completos, :]
        n = size(Xw, 1)
        if n < min_n_capital
            @printf("%-22s n=%d (abaixo do mínimo de %d observações estacionárias para calibrar cópula de %d dimensões, não calculado aqui)\n",
                    nome, n, min_n_capital, d)
            continue
        end
        T = kendall_tau_matrix(Xw)
        corr = tau_to_gaussian_corr(T; metodo=psd)
        Sgc = gaussian_copula_sum_sample(marg, corr, NSIM)
        es_gc = es_var(Sgc, ALPHA)
        frac = (es_gc.es - es_indep.es) / (es_comon.es - es_indep.es)
        @printf("%-22s n=%-4d VaR99=%.4f  ES99=%.4f  (fração indep->comon em ES: %.2f)\n",
                nome, n, es_gc.var, es_gc.es, frac)
        push!(resultados, (janela=nome, n=n, var99=es_gc.var, es99=es_gc.es))
    end

    if write_csv
        outpath = joinpath(OUT, "$(nome_trilha)_capital_agregado.csv")
        open(outpath, "w") do io
            println(io, "janela,n,var99,es99")
            for r in resultados
                println(io, join([r.janela, r.n, r.var99, r.es99], ","))
            end
        end
        @printf("Resultados exportados em %s\n", outpath)
    end
    return resultados
end

function main()
    Xc, setores_c, trimestres = load_credit()
    Xc_diff = first_differences(Xc)
    trimestres_diff = trimestres[2:end]
    janelas_credito = [("calma_2017_2019", "2017Q1", "2019Q4"),
                        ("estresse_2015_2016", "2015Q1", "2016Q4"),
                        ("estresse_2020", "2020Q1", "2020Q4")]
    # min_n_capital=12: com d=19 setores, exige ao menos uma janela do tamanho de
    # "calma" (3 anos) para calibrar a correlação com alguma confiança. As duas
    # janelas de estresse (n=8 e n=4 em diferenças) ficam só nas medidas de
    # dependência (dependence_credit.jl), não traduzidas em capital aqui.
    run_track("credito", Xc, Xc_diff, trimestres_diff, janelas_credito, fit_marginals_credit, 4000; min_n_capital=12)

    Xb, setores_b, datas = load_b3()
    janelas_b3 = [("calma_2017_2019", 20170101, 20191231),
                  ("estresse_2015_2016", 20150101, 20161231),
                  ("estresse_2020", 20200101, 20201231)]
    run_track("b3", Xb, Xb, datas, janelas_b3, fit_marginals_b3, 4000; min_n_capital=30)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
