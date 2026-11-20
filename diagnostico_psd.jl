# diagnostico_psd.jl
#
# Positividade da matriz sin(πτ/2) por janela e trilha (antes de qualquer
# correção), efeito da correção por corte de autovalores (versão anterior) e
# da projeção de Higham, e comparação de capital com equicorrelação calibrada
# no τ médio. Usa números aleatórios comuns (mesma semente antes de cada
# simulação) para que as diferenças de capital reflitam só a matriz.
using Printf, Random, Statistics, LinearAlgebra, Distributions
include(joinpath(@__DIR__, "aggregate_capital.jl"))

function ispd(M)
    try
        cholesky(Symmetric(Matrix(M))); return true
    catch
        return false
    end
end

function cap(marg, corr, es_indep, es_comon; seed=20261102)
    Random.seed!(seed)
    S = gaussian_copula_sum_sample(marg, corr, NSIM)
    e = es_var(S, ALPHA)
    return (var=e.var, es=e.es, frac=(e.es - es_indep.es) / (es_comon.es - es_indep.es))
end

function diag_track(nome, X_marg, X_corr, periodos, janelas, fit_fun, min_n)
    Random.seed!(20261102)
    marg = fit_fun(X_marg)
    d = length(marg)
    Sin = independent_sum_sample(marg, NSIM); Sco = comonotonic_sum_sample(marg, NSIM)
    ein = es_var(Sin, ALPHA); eco = es_var(Sco, ALPHA)
    rows = String[]
    for (jn, ini, fim) in janelas
        sel = periodos isa Vector{String} ? window_select(periodos, ini, fim) : [p >= ini && p <= fim for p in periodos]
        Xf = X_corr[sel, :]
        Xw = Xf[[all(!isnan, Xf[i, :]) for i in 1:size(Xf, 1)], :]
        n = size(Xw, 1); n < min_n && continue
        T = kendall_tau_matrix(Xw)
        R = tau_to_raw_corr(T)
        lam = eigvals(Symmetric(R))
        C = Matrix(nearest_psd(R)); H = nearest_corr_higham(R)
        E, tbar, rho = equicorr_from_tau(T)
        cc = cap(marg, C, ein, eco); ch = cap(marg, H, ein, eco); ce = cap(marg, E, ein, eco)
        @printf("\n[%s %s] n=%d d=%d  rank(dados)<=%d  tau_medio=%.4f rho_equi=%.4f\n", nome, jn, n, d, min(n - 1, d), tbar, rho)
        @printf("  menor autovalor sin(pi*tau/2) = %.6f ; autovalores negativos = %d ; Cholesky passa = %s\n", minimum(lam), count(<(0), lam), ispd(R))
        @printf("  clip : ||.||_F=%.4f  max|dij|=%.4f  menor autovalor=%.2e  Cholesky=%s\n", norm(C - R), maximum(abs.(C - R)), minimum(eigvals(Symmetric(C))), ispd(C))
        @printf("  Higham: ||.||_F=%.4f  max|dij|=%.4f  menor autovalor=%.2e  Cholesky=%s  diag max desvio=%.1e\n", norm(H - R), maximum(abs.(H - R)), minimum(eigvals(Symmetric(H))), ispd(H), maximum(abs.(diag(H) .- 1)))
        @printf("  ||clip-Higham||_F=%.4f\n", norm(C - H))
        @printf("  CRN  clip  : VaR=%.4f ES=%.4f frac=%.3f\n", cc.var, cc.es, cc.frac)
        @printf("  CRN  Higham: VaR=%.4f ES=%.4f frac=%.3f\n", ch.var, ch.es, ch.frac)
        @printf("  CRN  equi  : VaR=%.4f ES=%.4f frac=%.3f\n", ce.var, ce.es, ce.frac)
        push!(rows, join([nome, jn, n, d, minimum(lam), count(<(0), lam), ispd(R), norm(C - R), maximum(abs.(C - R)), norm(H - R), maximum(abs.(H - R)), cc.var, cc.es, cc.frac, ch.var, ch.es, ch.frac, ce.var, ce.es, ce.frac, tbar, rho], ","))
    end
    return rows
end

function main_diag()
    Xc, _, tri = load_credit(); Xd = first_differences(Xc)
    r1 = diag_track("credito", Xc, Xd, tri[2:end],
        [("calma_2017_2019", "2017Q1", "2019Q4"), ("estresse_2015_2016", "2015Q1", "2016Q4"), ("estresse_2020", "2020Q1", "2020Q4")],
        fit_marginals_credit, 12)
    Xb, _, datas = load_b3()
    r2 = diag_track("b3", Xb, Xb, datas,
        [("calma_2017_2019", 20170101, 20191231), ("estresse_2015_2016", 20150101, 20161231), ("estresse_2020", 20200101, 20201231)],
        fit_marginals_b3, 30)
    open(joinpath(OUT, "psd_diagnostico.csv"), "w") do io
        println(io, "trilha,janela,n,d,menor_autovalor_bruto,n_autovalores_negativos,cholesky_bruto,frob_clip,maxabs_clip,frob_higham,maxabs_higham,var_clip,es_clip,frac_clip,var_higham,es_higham,frac_higham,var_equi,es_equi,frac_equi,tau_medio,rho_equi")
        foreach(r -> println(io, r), vcat(r1, r2))
    end
end

main_diag()
