# aggregate_capital.jl
#
# Tradução das medidas de dependência em capital (ES/VaR a 99%) sobre
# distribuições CALIBRADAS, não sobre a perda realizada: marginais ajustadas
# na série completa de cada trilha (crédito, ações), combinadas com a
# estrutura de dependência (matriz de Kendall's tau -> cópula Gaussiana)
# estimada em cada janela, sob quatro hipóteses: independência, dependência
# calibrada, comonotonicidade (fórmula fechada via simulação com U comum) e
# pior caso de VaR via rearrangement algorithm dado só as marginais.
#
# A perda setorial agregada efetivamente observada em cada janela entra só
# como ilustração pontual, sobreposta aos quatro cenários, nunca como a
# estimativa central.

using Printf, Random, Statistics, LinearAlgebra, Distributions
include("src/Comonotonicidade.jl")
using .Comonotonicidade

const OUT = joinpath(@__DIR__, "data", "output")
const NSIM = 300_000
const ALPHA = 0.99

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
    trimestres = sort(unique(r.trimestre for r in rows))
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

"Projeta uma matriz simétrica para a mais próxima positiva semidefinida (clip de autovalores)."
function nearest_psd(M::AbstractMatrix{<:Real})
    S = Symmetric((M + M') / 2)
    F = eigen(S)
    vals = max.(F.values, 1e-8)
    return Symmetric(F.vectors * Diagonal(vals) * F.vectors')
end

"Converte matriz de Kendall's tau em matriz de correlação de cópula Gaussiana (ρ=sin(πτ/2)), corrigida para PSD."
function tau_to_gaussian_corr(T::AbstractMatrix{<:Real})
    d = size(T, 1)
    R = [i == j ? 1.0 : sin(pi * T[i, j] / 2) for i in 1:d, j in 1:d]
    return Matrix(nearest_psd(R))
end

"""
Para a trilha de crédito: marginal de 'perda' por setor = taxa de inadimplência
trimestral, ajustada a uma Gamma por método dos momentos na série completa.
Para a trilha de ações: marginal de 'perda' = retorno diário negado, ajustada
a uma Normal por método dos momentos na série completa.
"""
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

function fit_marginals_b3(X::AbstractMatrix{<:Real})
    d = size(X, 2)
    marg = Vector{Normal}(undef, d)
    for i in 1:d
        perdas = filter(!isnan, -X[:, i])
        marg[i] = Normal(mean(perdas), std(perdas))
    end
    return marg
end

function window_select(trimestres::Vector{String}, ini::String, fim::String)
    key(q) = parse(Int, q[1:4]) * 10 + parse(Int, q[6])
    return [key(t) >= key(ini) && key(t) <= key(fim) for t in trimestres]
end

function run_track(nome_trilha::AbstractString, X::Matrix{Float64}, setores::Vector{String},
                    periodos::Vector, janelas::Vector{<:Tuple}, fit_fun::Function, ra_N::Int)
    Random.seed!(20261102)
    marg = fit_fun(X)
    d = length(marg)
    @printf("\n=== %s (d=%d segmentos) ===\n", nome_trilha, d)

    Sindep = independent_sum_sample(marg, NSIM)
    Scomon = comonotonic_sum_sample(marg, NSIM)
    es_indep = es_var(Sindep, ALPHA)
    es_comon = es_var(Scomon, ALPHA)
    wv, _ = rearrange_worst_var(marg, ALPHA, ra_N)

    @printf("Independência:    VaR99=%.4f  ES99=%.4f\n", es_indep.var, es_indep.es)
    @printf("Comonotonicidade: VaR99=%.4f  ES99=%.4f\n", es_comon.var, es_comon.es)
    @printf("Pior caso (RA, N=%d): VaR99=%.4f  (razão sobre VaR comonotônico: %.2fx)\n",
            ra_N, wv, wv / es_comon.var)

    resultados = NamedTuple[]
    push!(resultados, (janela="independencia", n=NaN, var99=es_indep.var, es99=es_indep.es))
    push!(resultados, (janela="comonotonicidade", n=NaN, var99=es_comon.var, es99=es_comon.es))
    push!(resultados, (janela="pior_caso_RA", n=NaN, var99=wv, es99=NaN))

    for (nome, ini, fim) in janelas
        sel = if periodos isa Vector{String}
            window_select(periodos, ini, fim)
        else
            [p >= ini && p <= fim for p in periodos]
        end
        Xw_full = X[sel, :]
        completos = [all(!isnan, Xw_full[i, :]) for i in 1:size(Xw_full, 1)]
        Xw = Xw_full[completos, :]
        n = size(Xw, 1)
        if n < 8
            @printf("%-22s n=%d (amostra insuficiente para calibrar cópula, pulando)\n", nome, n)
            continue
        end
        T = kendall_tau_matrix(Xw)
        corr = tau_to_gaussian_corr(T)
        Sgc = gaussian_copula_sum_sample(marg, corr, NSIM)
        es_gc = es_var(Sgc, ALPHA)
        frac = (es_gc.es - es_indep.es) / (es_comon.es - es_indep.es)
        @printf("%-22s n=%-4d VaR99=%.4f  ES99=%.4f  (fração indep->comon em ES: %.2f)\n",
                nome, n, es_gc.var, es_gc.es, frac)
        push!(resultados, (janela=nome, n=n, var99=es_gc.var, es99=es_gc.es))
    end

    outpath = joinpath(OUT, "$(nome_trilha)_capital_agregado.csv")
    open(outpath, "w") do io
        println(io, "janela,n,var99,es99")
        for r in resultados
            println(io, join([r.janela, r.n, r.var99, r.es99], ","))
        end
    end
    @printf("Resultados exportados em %s\n", outpath)
    return resultados
end

function main()
    Xc, setores_c, trimestres = load_credit()
    janelas_credito = [("calma_2017_2019", "2017Q1", "2019Q4"),
                        ("estresse_2015_2016", "2015Q1", "2016Q4"),
                        ("estresse_2020", "2020Q1", "2020Q4")]
    run_track("credito", Xc, setores_c, trimestres, janelas_credito, fit_marginals_credit, 4000)

    Xb, setores_b, datas = load_b3()
    janelas_b3 = [("calma_2017_2019", 20170101, 20191231),
                  ("estresse_2015_2016", 20150101, 20161231),
                  ("estresse_2020", 20200101, 20201231)]
    run_track("b3", Xb, setores_b, datas, janelas_b3, fit_marginals_b3, 4000)
end

main()
