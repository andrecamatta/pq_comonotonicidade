# dependence_b3.jl
#
# Mede a dependência entre setores de ações da B3 (segunda aplicação), nas
# mesmas janelas de calendário usadas na trilha de crédito, com retornos
# diários (amostra muito maior que o painel trimestral de crédito).
#
# Diferente da taxa de inadimplência setorial (crédito), o retorno diário já
# é a variação período a período do preço (não o nível), portanto já é
# estacionário por construção. Nenhuma diferenciação adicional é aplicada
# aqui.

using Printf, Random, Statistics, Dates
include("src/Comonotonicidade.jl")
using .Comonotonicidade

const OUT = joinpath(@__DIR__, "data", "output")

function load_panel(path::AbstractString)
    lines = readlines(path)
    rows = NamedTuple[]
    for l in lines[2:end]
        cols = split(l, ',')
        push!(rows, (data = parse(Int, cols[1]), setor = replace(cols[2], "\"" => ""),
                      retorno = parse(Float64, cols[3])))
    end
    return rows
end

function pivot_matrix(rows, datas::Vector{Int}, setores::Vector{String})
    idx = Dict(d => i for (i, d) in enumerate(datas))
    jdx = Dict(s => j for (j, s) in enumerate(setores))
    X = fill(NaN, length(datas), length(setores))
    for r in rows
        haskey(idx, r.data) || continue
        haskey(jdx, r.setor) || continue
        X[idx[r.data], jdx[r.setor]] = r.retorno
    end
    return X
end

function main()
    Random.seed!(20261102)
    rows = load_panel(joinpath(OUT, "b3_painel_setorial.csv"))
    datas = sort(unique(r.data for r in rows))
    setor_count = Dict{String,Int}()
    for r in rows
        setor_count[r.setor] = get(setor_count, r.setor, 0) + 1
    end
    setores = sort(collect(keys(setor_count)))
    @printf("Painel B3: %d dias de calendário, %d setores\n", length(datas), length(setores))
    for s in setores
        @printf("  %-28s n=%d dias\n", s, setor_count[s])
    end
    println()

    Xraw = pivot_matrix(rows, datas, setores)

    janelas = [
        ("serie_completa", 20050101, 20251231),
        ("calma_2017_2019", 20170101, 20191231),
        ("estresse_2015_2016", 20150101, 20161231),
        ("estresse_2020", 20200101, 20201231),
    ]

    resultados = NamedTuple[]
    for (nome, ini, fim) in janelas
        sel = [d >= ini && d <= fim for d in datas]
        Xw_full = Xraw[sel, :]
        # exige dias com todos os setores observados (painel balanceado dentro da janela)
        completos = [all(!isnan, Xw_full[i, :]) for i in 1:size(Xw_full, 1)]
        Xw = Xw_full[completos, :]
        n = size(Xw, 1)
        if n < 30
            @printf("%-22s  n_dias=%3d  (amostra insuficiente, pulando)\n", nome, n)
            continue
        end
        tau = mean_kendall_tau(Xw)
        kappa = comonotonicity_coefficient(Xw)
        block_len = 10  # ~2 semanas de pregão, para capturar autocorrelação de curto prazo
        boot_tau = block_bootstrap_ci(Xw, mean_kendall_tau; nboot=2000, block_len=block_len)
        boot_kappa = block_bootstrap_ci(Xw, comonotonicity_coefficient; nboot=2000, block_len=block_len)
        @printf("%-22s  n_dias=%4d  [%d a %d]\n", nome, n, ini, fim)
        @printf("  tau médio = %.4f  IC90%% [%.4f, %.4f]\n", tau, boot_tau.ic_lo, boot_tau.ic_hi)
        @printf("  rho_c     = %.4f  IC90%% [%.4f, %.4f]\n", kappa, boot_kappa.ic_lo, boot_kappa.ic_hi)
        push!(resultados, (janela=nome, n=n, ini=ini, fim=fim, tau=tau, tau_lo=boot_tau.ic_lo, tau_hi=boot_tau.ic_hi,
                            kappa=kappa, kappa_lo=boot_kappa.ic_lo, kappa_hi=boot_kappa.ic_hi))

        if nome == "serie_completa"
            td = mean_tail_dependence(Xw; q=0.8)
            gc = gaussian_copula_tail_dependence(tau, 0.8, 2_000_000)
            lambda_indep = 1 - 0.8
            @printf("  dependência de cauda (q=0.8, %d pares): empírica lambda_U=%.4f lambda_L=%.4f | cópula Gaussiana (τ=%.4f) lambda_U=%.4f lambda_L=%.4f | independência=%.4f\n",
                    td.n_pairs, td.lambda_u, td.lambda_l, tau, gc.lambda_u, gc.lambda_l, lambda_indep)
            open(joinpath(OUT, "b3_dependencia_cauda.csv"), "w") do io
                println(io, "janela,q,lambda_u,lambda_l,n_pares,lambda_u_copula_gaussiana,lambda_l_copula_gaussiana,lambda_independencia")
                println(io, join(["serie_completa", 0.8, td.lambda_u, td.lambda_l, td.n_pairs, gc.lambda_u, gc.lambda_l, lambda_indep], ","))
            end
        end
        println()
    end

    outpath = joinpath(OUT, "b3_dependencia_janelas.csv")
    open(outpath, "w") do io
        println(io, "janela,n_dias,inicio,fim,tau_medio,tau_ic90_lo,tau_ic90_hi,rho_c,rho_c_ic90_lo,rho_c_ic90_hi")
        for r in resultados
            println(io, join([r.janela, r.n, r.ini, r.fim, r.tau, r.tau_lo, r.tau_hi, r.kappa, r.kappa_lo, r.kappa_hi], ","))
        end
    end
    @printf("Resultados exportados em %s\n", outpath)
end

main()
