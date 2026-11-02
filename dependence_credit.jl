# dependence_credit.jl
#
# Mede a dependência entre setores de crédito (SCR.data) por janela
# (estresse 2015-2016, estresse 2020, calma 2017-2019, série completa
# 2012Q3-2025Q4), com Kendall's tau médio, a medida de dependência ρ_c
# (DHAENE; LINDERS; SCHOUTENS; VYNCKE, 2014) e, na série completa, a
# dependência de cauda empírica, com intervalos de confiança por bootstrap
# em bloco temporal.
#
# Transformação para estacionariedade: a taxa de inadimplência setorial tem
# NÍVEL com tendência comum entre setores (ex. queda conjunta em 2020), o que
# infla qualquer medida de dependência calculada sobre o nível em vez da
# variação período a período. Todas as medidas de dependência deste script
# são calculadas em primeiras diferenças (Δtaxa_t = taxa_t - taxa_{t-1}),
# não no nível. Declara o número de observações de cada janela ao lado de
# cada estimativa, e suprime bootstrap quando a janela não comporta
# inferência (n < 8).

using Printf, Random, Statistics
include("src/Comonotonicidade.jl")
using .Comonotonicidade

const OUT = joinpath(@__DIR__, "data", "output")
const MIN_N_BOOTSTRAP = 8  # abaixo disso, bootstrap em bloco não é informativo (poucas configurações distintas)
const BLOCK_LEN = 4        # um ano de trimestres, fixo para todas as janelas que comportam inferência

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

    Xniv = pivot_matrix(rows, trimestres, setores)
    @assert all(!isnan, Xniv) "painel balanceado deveria estar completo"

    # Primeiras diferenças, rotuladas pelo trimestre em que a diferença TERMINA
    # (Δ em "2015Q1" = taxa(2015Q1) - taxa(2014Q4)), preservando o painel
    # balanceado de setores.
    Xdiff = first_differences(Xniv)
    trimestres_diff = trimestres[2:end]
    @printf("Painel em primeiras diferenças: %d observações (%s a %s)\n\n",
            length(trimestres_diff), trimestres_diff[1], trimestres_diff[end])

    janelas = [
        ("serie_completa", trimestres_diff[1], trimestres_diff[end]),
        ("calma_2017_2019", "2017Q1", "2019Q4"),
        ("estresse_2015_2016", "2015Q1", "2016Q4"),
        ("estresse_2020", "2020Q1", "2020Q4"),
    ]

    resultados = NamedTuple[]
    for (nome, ini, fim) in janelas
        sel = [quarter_sort_key(t) >= quarter_sort_key(ini) && quarter_sort_key(t) <= quarter_sort_key(fim)
               for t in trimestres_diff]
        Xw = Xdiff[sel, :]
        n = size(Xw, 1)
        tau = mean_kendall_tau(Xw)
        kappa = comonotonicity_coefficient(Xw)
        @printf("%-22s  n_diferencas=%2d  [%s a %s]\n", nome, n, ini, fim)
        if n >= MIN_N_BOOTSTRAP
            boot_tau = block_bootstrap_ci(Xw, mean_kendall_tau; nboot=3000, block_len=BLOCK_LEN)
            boot_kappa = block_bootstrap_ci(Xw, comonotonicity_coefficient; nboot=3000, block_len=BLOCK_LEN)
            @printf("  tau médio = %.4f  IC90%% [%.4f, %.4f]  (bloco=%d trimestres)\n",
                    tau, boot_tau.ic_lo, boot_tau.ic_hi, BLOCK_LEN)
            @printf("  rho_c     = %.4f  IC90%% [%.4f, %.4f]  (bloco=%d trimestres)\n",
                    kappa, boot_kappa.ic_lo, boot_kappa.ic_hi, BLOCK_LEN)
            push!(resultados, (janela=nome, n=n, ini=ini, fim=fim, tau=tau,
                                tau_lo=boot_tau.ic_lo, tau_hi=boot_tau.ic_hi,
                                kappa=kappa, kappa_lo=boot_kappa.ic_lo, kappa_hi=boot_kappa.ic_hi,
                                bloco=BLOCK_LEN))
        else
            @printf("  tau médio = %.4f  (n=%d < %d, inferência por bootstrap não reportada)\n", tau, n, MIN_N_BOOTSTRAP)
            @printf("  rho_c     = %.4f  (n=%d < %d, inferência por bootstrap não reportada)\n", kappa, n, MIN_N_BOOTSTRAP)
            push!(resultados, (janela=nome, n=n, ini=ini, fim=fim, tau=tau,
                                tau_lo=NaN, tau_hi=NaN, kappa=kappa, kappa_lo=NaN, kappa_hi=NaN,
                                bloco=0))
        end
        println()
    end

    outpath = joinpath(OUT, "credito_dependencia_janelas.csv")
    open(outpath, "w") do io
        println(io, "janela,n_diferencas,inicio,fim,tau_medio,tau_ic90_lo,tau_ic90_hi,rho_c,rho_c_ic90_lo,rho_c_ic90_hi,bloco")
        for r in resultados
            println(io, join([r.janela, r.n, r.ini, r.fim, r.tau, r.tau_lo, r.tau_hi, r.kappa, r.kappa_lo, r.kappa_hi, r.bloco], ","))
        end
    end
    @printf("Resultados exportados em %s\n", outpath)

    # Dependência de cauda empírica, só na série completa (única janela com n suficiente),
    # comparada com a cauda implícita de uma cópula Gaussiana calibrada no mesmo tau médio
    # e com a referência teórica de independência (= 1-q, exatamente, para marginais contínuas).
    td = mean_tail_dependence(Xdiff; q=0.8)
    tau_full = mean_kendall_tau(Xdiff)
    gc = gaussian_copula_tail_dependence(tau_full, 0.8, 2_000_000)
    lambda_indep = 1 - 0.8
    @printf("\nDependência de cauda (série completa, limiar q=0.8, %d pares de setores):\n", td.n_pairs)
    @printf("  empírica:              lambda_U = %.4f  lambda_L = %.4f\n", td.lambda_u, td.lambda_l)
    @printf("  cópula Gaussiana (τ=%.4f): lambda_U = %.4f  lambda_L = %.4f\n", tau_full, gc.lambda_u, gc.lambda_l)
    @printf("  independência (referência teórica, 1-q): lambda = %.4f\n", lambda_indep)
    open(joinpath(OUT, "credito_dependencia_cauda.csv"), "w") do io
        println(io, "janela,q,lambda_u,lambda_l,n_pares,lambda_u_copula_gaussiana,lambda_l_copula_gaussiana,lambda_independencia")
        println(io, join(["serie_completa", 0.8, td.lambda_u, td.lambda_l, td.n_pairs, gc.lambda_u, gc.lambda_l, lambda_indep], ","))
    end

    # Matriz de Kendall's tau da série completa (em diferenças), para referência de dispersão par a par
    T = kendall_tau_matrix(Xdiff)
    offdiag = [T[i,j] for i in 1:size(T,1) for j in 1:size(T,1) if i != j]
    @printf("\nMatriz de tau em diferenças (série completa): min=%.3f, mediana=%.3f, max=%.3f\n",
            minimum(offdiag), median(offdiag), maximum(offdiag))
end

main()
