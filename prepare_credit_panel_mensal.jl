# prepare_credit_panel_mensal.jl
#
# Painel MENSAL de inadimplência setorial do SCR.data (mesmo recorte do painel
# trimestral: PJ, modalidade Empréstimos, agregado por CNAE, células com menos
# de 30 operações descartadas). O SCR.data é publicado todo mês, e o painel
# trimestral de `prepare_credit_panel.jl` só usa os fins de trimestre, o que
# deixa 12 diferenças na janela calma. Aqui entram os 12 arquivos de cada ano.
#
# O rótulo do período é AAAA-MM, sem agrupar meses em trimestres (a função
# `quarter_label` do painel trimestral rotula como Q4 todo mês que não seja
# 3, 6 ou 9, e produziria datas duplicadas com 12 arquivos por ano). Duas
# observações com o mesmo rótulo são erro.
#
# Reproduzir:
#   ./extract_scr_mensal.sh <pasta com scrdata_AAAA.zip> <pasta mensal>
#   SCR_MENSAL_DIR=<pasta mensal> julia --project=. prepare_credit_panel_mensal.jl
using Printf
include(joinpath(@__DIR__, "prepare_credit_panel.jl"))  # aggregate_by_cnae, MIN_OPERACOES, OUT

const SCR_MENSAL_DIR = get(ENV, "SCR_MENSAL_DIR", joinpath(@__DIR__, "..", "..", "dados_pq", "scr", "mensal"))

"Rótulo AAAA-MM a partir do nome scrdata_AAAAMM.csv."
month_label(mes::AbstractString) = "$(mes[1:4])-$(mes[5:6])"

"Erro se algum rótulo de período aparece mais de uma vez."
function assert_unique_labels(labels)
    seen = Set{String}()
    for l in labels
        l in seen && error("data duplicada no painel: $l")
        push!(seen, l)
    end
    return true
end

function main_mensal()
    arquivos = sort(filter(f -> occursin(r"^scrdata_\d{6}\.csv$", f), readdir(SCR_MENSAL_DIR)))
    isempty(arquivos) && error("Nenhum arquivo scrdata_AAAAMM.csv em $SCR_MENSAL_DIR")
    rotulos = [month_label(match(r"scrdata_(\d{6})\.csv", a).captures[1]) for a in arquivos]
    assert_unique_labels(rotulos)
    @printf("Lendo %d arquivos mensais de %s (%s a %s)\n\n", length(arquivos), SCR_MENSAL_DIR, first(rotulos), last(rotulos))

    rows = NamedTuple[]
    for (arq, rot) in zip(arquivos, rotulos)
        agg = aggregate_by_cnae(joinpath(SCR_MENSAL_DIR, arq))
        for (cnae, v) in agg
            v.n_ops < MIN_OPERACOES && continue
            push!(rows, (mes = rot, cnae = cnae, n_operacoes = v.n_ops, carteira_ativa = v.carteira_ativa,
                         vencido_90 = v.vencido90, taxa_inadimplencia = v.vencido90 / v.carteira_ativa))
        end
    end
    assert_unique_labels(["$(r.mes)|$(r.cnae)" for r in rows])

    outpath = joinpath(OUT, "scr_painel_mensal.csv")
    open(outpath, "w") do io
        println(io, "mes,cnae,n_operacoes,carteira_ativa,vencido_90,taxa_inadimplencia")
        for r in rows
            println(io, join([r.mes, "\"$(r.cnae)\"", r.n_operacoes, r.carteira_ativa, r.vencido_90, r.taxa_inadimplencia], ","))
        end
    end
    contagem = Dict{String,Int}()
    for r in rows; contagem[r.cnae] = get(contagem, r.cnae, 0) + 1; end
    nmes = length(rotulos)
    bal = sort([s for (s, c) in contagem if c == nmes])
    @printf("\n%d linhas (mês x CNAE) em %s\n%d setores no total, %d presentes em todos os %d meses\n",
            length(rows), outpath, length(contagem), length(bal), nmes)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_mensal()
end
