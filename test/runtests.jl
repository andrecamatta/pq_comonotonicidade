# test/runtests.jl
#
# Testes de unidade do módulo Comonotonicidade.jl, em casos com resultado
# conhecido analiticamente ou por construção: coeficiente de dependência ρc
# em dados comonotônicos e independentes, limites teóricos do rearrangement
# algorithm (RA), primeiras diferenças, e dependência de cauda em dados
# comonotônicos. Reproduzir: `julia --project=. -e 'using Pkg; Pkg.test()'`.

using Test, Random, Statistics, Distributions
include(joinpath(@__DIR__, "..", "src", "Comonotonicidade.jl"))
using .Comonotonicidade

@testset "Comonotonicidade.jl" begin

    @testset "ρc em construção comonotônica exata" begin
        # Colunas são transformações afins crescentes do mesmo vetor base:
        # por construção, comonotônicas, correlação par a par = 1.
        Random.seed!(1)
        u = randn(500)
        X = hcat(u, 2.0 .* u .+ 3.0, 0.5 .* u .- 1.0, 4.0 .* u)
        @test comonotonicity_coefficient(X) ≈ 1.0 atol=1e-8
        @test mean_kendall_tau(X) ≈ 1.0 atol=1e-8
    end

    @testset "ρc perto de 0 sob independência" begin
        Random.seed!(2)
        X = randn(200_000, 4)  # colunas independentes por construção
        @test abs(comonotonicity_coefficient(X)) < 0.02
        @test abs(mean_kendall_tau(X)) < 0.02
    end

    @testset "ρc negativo sob dependência negativa" begin
        Random.seed!(3)
        u = randn(2000)
        X = hcat(u, -u)  # contramonotônico exato em d=2: Var(S) = 0 em toda observação
        # ρc não é necessariamente exatamente -1 com a fórmula de Dhaene et al. (o
        # denominador usa a variância do rearranjo comonotônico empírico, não a
        # aproximação (Σσᵢ)² de correlação linear perfeita), mas fica fortemente
        # negativo, já que a soma observada tem variância zero.
        @test comonotonicity_coefficient(X) < -0.9
    end

    @testset "Soma comonotônica: aditividade de quantis (fórmula fechada)" begin
        marg = [Gamma(2.0, 1.0), Gamma(3.0, 1.5), Exponential(2.0)]
        Scomon = comonotonic_sum_sample(marg, 500_000)
        alpha = 0.95
        var_soma = quantile(Scomon, alpha)
        var_marginal_somado = sum(quantile(m, alpha) for m in marg)
        @test isapprox(var_soma, var_marginal_somado; rtol=0.01)
    end

    @testset "Rearrangement algorithm: VaR(S) ≤ ES(S) ≤ Σ ES marginal" begin
        Random.seed!(4)
        marg = [Gamma(2.0, 1.0) for _ in 1:5]
        alpha = 0.95
        Scomon = comonotonic_sum_sample(marg, 500_000)
        var_comon = quantile(Scomon, alpha)
        es_comon = mean(Scomon[Scomon .>= var_comon])  # = Σ ES marginal, teto teórico
        pior_var, _ = rearrange_worst_var(marg, alpha, 3000)
        # O pior VaR nunca pode superar o teto teórico Σ ES marginal (= ES comonotônico).
        @test pior_var <= es_comon * 1.01
        # E, nesta classe de marginais homogêneas, fica igual ou acima do VaR comonotônico,
        # a motivação central do artigo: a soma de VaRs não é garantidamente o pior caso.
        @test pior_var >= var_comon * 0.99
    end

    @testset "Primeiras diferenças" begin
        X = Float64[1 10; 2 20; 4 40; 7 70]
        D = first_differences(X)
        @test size(D) == (3, 2)
        @test D == Float64[1 10; 2 20; 3 30]
    end

    @testset "Dependência de cauda empírica em dados comonotônicos" begin
        Random.seed!(5)
        u = rand(2000)
        x = quantile.(Normal(), u)
        y = quantile.(Exponential(1.0), u)  # comonotônico com x, marginal diferente
        r = empirical_tail_dependence(x, y; q=0.9)
        @test r.lambda_u > 0.9
        @test r.lambda_l > 0.9
    end

    @testset "block_bootstrap_ci: intervalo cobre a estimativa pontual" begin
        Random.seed!(6)
        X = randn(60, 3)
        b = block_bootstrap_ci(X, mean_kendall_tau; nboot=500, block_len=4)
        @test b.ic_lo <= b.ponto <= b.ic_hi
    end

    @testset "comonotonic_rearrangement: cada coluna sai ordenada" begin
        Random.seed!(7)
        X = randn(50, 3)
        Xc = comonotonic_rearrangement(X)
        @test all(issorted(Xc[:, j]) for j in 1:3)
        # mesmos valores por coluna, só reordenados
        @test all(sort(X[:, j]) == Xc[:, j] for j in 1:3)
    end

    @testset "gaussian_copula_tail_dependence: independência dá λ≈1-q" begin
        Random.seed!(8)
        r = gaussian_copula_tail_dependence(0.0, 0.8, 500_000)
        @test isapprox(r.lambda_u, 0.2; atol=0.01)
        @test isapprox(r.lambda_l, 0.2; atol=0.01)
    end

end

# Funções de calibração e de painel dos scripts do crédito (painel mensal, projeção de Higham, escolha da correção).
# Os scripts reincluem o módulo, o que só gera um aviso de redefinição.
include(joinpath(@__DIR__, "..", "aggregate_capital.jl"))
include(joinpath(@__DIR__, "..", "prepare_credit_panel_mensal.jl"))

@testset "Painel mensal e calibração da cópula" begin
    @testset "rótulo AAAA-MM e datas duplicadas" begin
        @test month_label("201712") == "2017-12"
        @test month_label("201201") == "2012-01"
        @test assert_unique_labels(["2017-01", "2017-02", "2017-03"])
        @test_throws ErrorException assert_unique_labels(["2017-01", "2017-02", "2017-01"])
        # quarter_label (painel trimestral) colapsa meses em Q4, por isso não serve para 12 arquivos por ano
        @test quarter_label("201710") == quarter_label("201711") == quarter_label("201712")
    end

    @testset "window_select em rótulos mensais e trimestrais" begin
        meses = ["2016-12", "2017-01", "2019-12", "2020-01"]
        @test window_select(meses, "2017-01", "2019-12") == [false, true, true, false]
        tri = ["2016Q4", "2017Q1", "2019Q4", "2020Q1"]
        @test window_select(tri, "2017Q1", "2019Q4") == [false, true, true, false]
    end

    @testset "Higham: correlação PD inalterada, indefinida vira PD com diagonal 1" begin
        Random.seed!(11)
        A = randn(30, 5); R = cor(A)
        H = nearest_corr_higham(R)
        @test maximum(abs.(H - R)) < 1e-6
        # τ de 4 observações para 6 variáveis: posto baixo, sin(πτ/2) indefinida
        X = randn(4, 6)
        T = Comonotonicidade.kendall_tau_matrix(X)
        Rraw = tau_to_raw_corr(T)
        Hh = nearest_corr_higham(Rraw)
        @test cholesky_ok(Hh)
        @test all(isapprox.(diag(Hh), 1.0; atol=1e-10))
        @test issymmetric(Hh)
    end

    @testset ":auto só corrige quando o Cholesky falha" begin
        Random.seed!(12)
        Xpd = randn(200, 4)
        Tpd = Comonotonicidade.kendall_tau_matrix(Xpd)
        @test cholesky_ok(tau_to_raw_corr(Tpd))
        @test tau_to_gaussian_corr(Tpd; metodo=:auto) == tau_to_raw_corr(Tpd)
        Xbad = randn(5, 9)
        Tbad = Comonotonicidade.kendall_tau_matrix(Xbad)
        if !cholesky_ok(tau_to_raw_corr(Tbad))
            @test tau_to_gaussian_corr(Tbad; metodo=:auto) == tau_to_gaussian_corr(Tbad; metodo=:higham)
            @test cholesky_ok(tau_to_gaussian_corr(Tbad; metodo=:auto))
        end
    end

    @testset "equicorrelação: τ médio e ρ=sin(πτ̄/2)" begin
        T = [1.0 0.2 0.4; 0.2 1.0 0.6; 0.4 0.6 1.0]
        E, tbar, rho = equicorr_from_tau(T)
        @test tbar ≈ 0.4
        @test rho ≈ sin(pi * 0.4 / 2)
        @test all(E[i, j] == rho for i in 1:3 for j in 1:3 if i != j)
        @test cholesky_ok(E)
    end
end
