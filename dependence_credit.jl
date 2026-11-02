# dependence_credit.jl
#
# Mede a dependência entre setores de crédito (SCR.data) por janela
# (estresse 2015-2016, estresse 2020, calma 2017-2019, série completa
# 2012Q3-2025Q4), com Kendall's tau médio e coeficiente de comonotonicidade
# κ, e intervalos de confiança por bootstrap em bloco temporal. Declara o
# número de trimestres de cada janela ao lado de cada estimativa.

using Printf, Random, Statistics
include("src/Comonotonicidade.jl")
using .Comonotonicidade

const OUT = joinpath(@__DIR__, "data", "output")

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

function load_panel(path::AbstractString)
    lines = readlines(path)
    rows = NamedTuple[]
    for l in lines[2:end]
        cols = split_csv_line(l)
        push!(rows, (trimestre = cols[1], mes = cols[2], cnae = cols[3],
                      taxa = parse(Float64, cols[7])))
    end
    return rows
end

"Pivota para matriz trimestre x setor (só setores balanceados em todos os trimestres do painel completo)."
function pivot_matrix(rows, trimestres::AbstractVector, setores::AbstractVector)
    idx = Dict(t => i for (i, t) in enumerate(trimestres))
    jdx = Dict(s => j for (j, s) in enumerate(setores))
    X = fill(NaN, length(trimestres), length(setores))
    for r in rows
        haskey(idx, r.trimestre) || continue
        haskey(jdx, r.cnae) || continue
        X[idx[r.trimestre], jdx[r.cnae]] = r.taxa
    end
    return X
end

function quarter_sort_key(q::AbstractString)
    ano = parse(Int, q[1:4])
    tri = parse(Int, q[6])
    return ano * 10 + tri
end

function main()
    Random.seed!(20261102)
    rows = load_panel(joinpath(OUT, "scr_painel_setorial.csv"))
    trimestres = sort(unique(r.trimestre for r in rows); by=quarter_sort_key)
    cnae_count = Dict{String,Int}()
    for r in rows
        cnae_count[r.cnae] = get(cnae_count, r.cnae, 0) + 1
    end
    setores = sort([s for (s, c) in cnae_count if c == length(trimestres)])
    @printf("Painel: %d trimestres, %d setores balanceados\n\n", length(trimestres), length(setores))

    Xfull = pivot_matrix(rows, trimestres, setores)
    @assert all(!isnan, Xfull) "painel balanceado deveria estar completo"

    janelas = [
        ("serie_completa", trimestres[1], trimestres[end]),
        ("calma_2017_2019", "2017Q1", "2019Q4"),
        ("estresse_2015_2016", "2015Q1", "2016Q4"),
        ("estresse_2020", "2020Q1", "2020Q4"),
    ]

    resultados = NamedTuple[]
    for (nome, ini, fim) in janelas
        sel = [quarter_sort_key(t) >= quarter_sort_key(ini) && quarter_sort_key(t) <= quarter_sort_key(fim)
               for t in trimestres]
        Xw = Xfull[sel, :]
        n = size(Xw, 1)
        tau = mean_kendall_tau(Xw)
        kappa = comonotonicity_coefficient(Xw)
        block_len = max(2, min(4, n ÷ 3))
        boot_tau = n >= 4 ? block_bootstrap_ci(Xw, mean_kendall_tau; nboot=3000, block_len=block_len) : nothing
        boot_kappa = n >= 4 ? block_bootstrap_ci(Xw, comonotonicity_coefficient; nboot=3000, block_len=block_len) : nothing
        @printf("%-22s  n_trimestres=%2d  [%s a %s]\n", nome, n, ini, fim)
        @printf("  tau médio = %.4f", tau)
        if !isnothing(boot_tau)
            @printf("  IC90%% [%.4f, %.4f]\n", boot_tau.ic_lo, boot_tau.ic_hi)
        else
            println("  (n insuficiente para bootstrap)")
        end
        @printf("  kappa     = %.4f", kappa)
        if !isnothing(boot_kappa)
            @printf("  IC90%% [%.4f, %.4f]\n", boot_kappa.ic_lo, boot_kappa.ic_hi)
        else
            println("  (n insuficiente para bootstrap)")
        end
        push!(resultados, (janela=nome, n=n, ini=ini, fim=fim, tau=tau,
                            tau_lo=isnothing(boot_tau) ? NaN : boot_tau.ic_lo,
                            tau_hi=isnothing(boot_tau) ? NaN : boot_tau.ic_hi,
                            kappa=kappa,
                            kappa_lo=isnothing(boot_kappa) ? NaN : boot_kappa.ic_lo,
                            kappa_hi=isnothing(boot_kappa) ? NaN : boot_kappa.ic_hi))
        println()
    end

    outpath = joinpath(OUT, "credito_dependencia_janelas.csv")
    open(outpath, "w") do io
        println(io, "janela,n_trimestres,inicio,fim,tau_medio,tau_ic90_lo,tau_ic90_hi,kappa,kappa_ic90_lo,kappa_ic90_hi")
        for r in resultados
            println(io, join([r.janela, r.n, r.ini, r.fim, r.tau, r.tau_lo, r.tau_hi, r.kappa, r.kappa_lo, r.kappa_hi], ","))
        end
    end
    @printf("Resultados exportados em %s\n", outpath)

    # Matriz de Kendall's tau da série completa, para referência de dispersão par a par
    T = kendall_tau_matrix(Xfull)
    offdiag = [T[i,j] for i in 1:size(T,1) for j in 1:size(T,1) if i != j]
    @printf("\nMatriz de tau (série completa): min=%.3f, mediana=%.3f, max=%.3f\n",
            minimum(offdiag), median(offdiag), maximum(offdiag))
end

main()
