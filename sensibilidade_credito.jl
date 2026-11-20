# sensibilidade_credito.jl
#
# Sensibilidade da calibração da cópula de crédito à frequência das diferenças.
# O painel mensal permite formar diferenças TRIMESTRAIS NÃO SOBREPOSTAS em três
# fases (níveis de fim de trimestre em mar/jun/set/dez, em jan/abr/jul/out ou em
# fev/mai/ago/nov). A fase 0 reproduz o painel trimestral usado até a versão
# anterior (12 diferenças na janela calma). Marginais Gama, horizonte do capital e
# semente são os de aggregate_capital.jl; a mesma semente abre cada simulação
# (números aleatórios comuns), então as diferenças refletem só a matriz de dependência.
#
# Saída: data/output/credito_sensibilidade.csv e na tela.
using Printf, Random, Statistics, LinearAlgebra, Distributions
include(joinpath(@__DIR__, "aggregate_capital.jl"))

function capital_sim(marg, corr, ein, eco)
    Random.seed!(20261102)
    e = es_var(gaussian_copula_sum_sample(marg, corr, NSIM), ALPHA)
    return (var=e.var, es=e.es, frac=(e.es - ein.es) / (eco.es - ein.es))
end

function main_sens()
    Xc, setores, _ = load_credit()
    Xm, meses = load_credit_mensal(setores)
    Random.seed!(20261102)
    marg = fit_marginals_credit(Xc)
    ein = es_var(independent_sum_sample(marg, NSIM), ALPHA)
    eco = es_var(comonotonic_sum_sample(marg, NSIM), ALPHA)
    ini, fim = "2017-01", "2019-12"
    linhas = String[]
    fases = [(0, "fins de trimestre (mar, jun, set, dez)"), (1, "jan, abr, jul, out"), (2, "fev, mai, ago, nov")]
    for (fase, desc) in fases
        # níveis nos meses m com (m - 3 + fase) % 3 == 0 ... fase 0: m ∈ {3,6,9,12}; fase 1: {1,4,7,10}; fase 2: {2,5,8,11}
        alvo = fase == 0 ? 0 : (fase == 1 ? 1 : 2)
        idx = [i for (i, m) in enumerate(meses) if parse(Int, m[6:7]) % 3 == alvo]
        Xq = Xm[idx, :]; per = meses[idx]
        Xd = first_differences(Xq); pd = per[2:end]
        sel = [t >= ini && t <= fim for t in pd]
        Xw = Xd[sel, :]; n = size(Xw, 1)
        T = kendall_tau_matrix(Xw)
        R = tau_to_raw_corr(T); lam = minimum(eigvals(Symmetric(R))); ok = cholesky_ok(R)
        C = tau_to_gaussian_corr(T; metodo=:auto)
        E, tbar, rho = equicorr_from_tau(T)
        cg = capital_sim(marg, C, ein, eco); ce = capital_sim(marg, E, ein, eco)
        @printf("fase %d (%s): n=%d tau_medio=%.4f menor autovalor=%+.4f Cholesky=%s | Higham/auto VaR=%.4f ES=%.4f frac=%.3f | equicorr VaR=%.4f ES=%.4f frac=%.3f\n",
                fase, desc, n, tbar, lam, ok, cg.var, cg.es, cg.frac, ce.var, ce.es, ce.frac)
        push!(linhas, join(["nao_sobrepostas_fase$fase", n, tbar, lam, ok, cg.var, cg.es, cg.frac, ce.var, ce.es, ce.frac], ","))
    end
    # Mensal (36 diferenças) com os mesmos números aleatórios, para a comparação direta
    Xd = first_differences(Xm); pd = meses[2:end]
    Xw = Xd[[t >= ini && t <= fim for t in pd], :]; n = size(Xw, 1)
    T = kendall_tau_matrix(Xw); R = tau_to_raw_corr(T)
    lam = minimum(eigvals(Symmetric(R))); ok = cholesky_ok(R)
    C = tau_to_gaussian_corr(T; metodo=:auto); E, tbar, rho = equicorr_from_tau(T)
    cg = capital_sim(marg, C, ein, eco); ce = capital_sim(marg, E, ein, eco)
    @printf("mensal: n=%d tau_medio=%.4f menor autovalor=%+.4f Cholesky=%s | VaR=%.4f ES=%.4f frac=%.3f | equicorr VaR=%.4f ES=%.4f frac=%.3f\n",
            n, tbar, lam, ok, cg.var, cg.es, cg.frac, ce.var, ce.es, ce.frac)
    push!(linhas, join(["mensal", n, tbar, lam, ok, cg.var, cg.es, cg.frac, ce.var, ce.es, ce.frac], ","))
    open(joinpath(OUT, "credito_sensibilidade.csv"), "w") do io
        println(io, "diferencas,n,tau_medio,menor_autovalor,cholesky_bruto,var99,es99,fracao,var99_equi,es99_equi,fracao_equi")
        foreach(l -> println(io, l), linhas)
    end
end

main_sens()
