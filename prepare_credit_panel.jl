# prepare_credit_panel.jl
#
# Constrói o painel setorial completo do SCR.data (Banco Central) usado na
# Parte 4 da série: taxa de inadimplência (vencido > 90 dias / carteira ativa)
# por setor CNAE e por trimestre, de 2012Q3 a 2025Q4 (54 trimestres).
#
# Reaproveita o mesmo recorte da Parte 2 (pq_panjer): cliente = PJ (pessoa
# jurídica), modalidade = "Empréstimos", células agregadas por CNAE (todas as
# UF e portes), excluindo células suprimidas por sigilo bancário
# (numero_de_operacoes < 0, limite de 15 operações da metodologia do BCB).
# Os arquivos brutos (CSV dentro de ZIP anual do BCB) já estão em disco,
# baixados para a Parte 2, e não são copiados nem versionados aqui.
#
# Fonte: SCR.data, dadosabertos.bcb.gov.br. Reproduzir: apontar
# PANJER_RAW_DIR para a pasta data/raw/ do repositório pq_panjer (Parte 2) e
# rodar `julia --project=. prepare_credit_panel.jl`.

using Printf

const PANJER_RAW_DIR = get(ENV, "PANJER_RAW_DIR",
    joinpath(@__DIR__, "..", "..", "panjer", "pq_panjer", "data", "raw"))
const OUT = joinpath(@__DIR__, "data", "output")
mkpath(OUT)

const MIN_OPERACOES = 30  # amostra mínima por célula setor-trimestre (mesmo corte da Parte 2)

"Converte string BR ('1.234,56' ou '1234,56') para Float64."
function parse_brnum(s::AbstractString)
    s = replace(s, "\"" => "")
    s = replace(s, "." => "")
    s = replace(s, "," => ".")
    return parse(Float64, s)
end

"Lê um CSV trimestral do SCR.data e agrega por CNAE (campo 5), para cliente=PJ e modalidade=Empréstimos."
function aggregate_by_cnae(path::AbstractString)
    agg = Dict{String,NamedTuple{(:n_ops, :vencido90, :carteira_ativa),Tuple{Float64,Float64,Float64}}}()
    n_lido = 0
    n_usado = 0
    open(path, "r") do io
        readline(io)  # descarta cabeçalho
        for line in eachline(io)
            n_lido += 1
            cols = split(line, ';')
            length(cols) < 23 && continue
            cli = replace(cols[4], "\"" => "")
            mod = replace(cols[7], "\"" => "")
            (cli == "PJ" && mod == "Empréstimos") || continue
            cnae = replace(cols[5], "\"" => "")
            n_ops = parse(Float64, replace(cols[11], "\"" => ""))
            n_ops < 0 && continue  # célula suprimida por sigilo
            vencido90 = parse_brnum(cols[20])
            carteira_ativa = parse_brnum(cols[22])
            carteira_ativa <= 0 && continue
            n_usado += 1
            prev = get(agg, cnae, (n_ops = 0.0, vencido90 = 0.0, carteira_ativa = 0.0))
            agg[cnae] = (n_ops = prev.n_ops + n_ops,
                         vencido90 = prev.vencido90 + vencido90,
                         carteira_ativa = prev.carteira_ativa + carteira_ativa)
        end
    end
    @printf("  %s: %d linhas lidas, %d usadas\n", basename(path), n_lido, n_usado)
    return agg
end

function quarter_label(mes::AbstractString)
    ano = mes[1:4]
    m = parse(Int, mes[5:6])
    tri = m == 3 ? 1 : m == 6 ? 2 : m == 9 ? 3 : 4
    return "$(ano)Q$(tri)"
end

function main()
    arquivos = sort(filter(f -> occursin(r"^scrdata_\d{6}\.csv$", f), readdir(PANJER_RAW_DIR)))
    isempty(arquivos) && error("Nenhum arquivo scrdata_*.csv encontrado em $PANJER_RAW_DIR")
    @printf("Lendo %d arquivos trimestrais de %s\n\n", length(arquivos), PANJER_RAW_DIR)

    rows = NamedTuple[]
    for arq in arquivos
        mes = match(r"scrdata_(\d{6})\.csv", arq).captures[1]
        tri = quarter_label(mes)
        agg = aggregate_by_cnae(joinpath(PANJER_RAW_DIR, arq))
        for (cnae, v) in agg
            v.n_ops < MIN_OPERACOES && continue
            taxa = v.vencido90 / v.carteira_ativa
            push!(rows, (trimestre = tri, mes = mes, cnae = cnae, n_operacoes = v.n_ops,
                         carteira_ativa = v.carteira_ativa, vencido_90 = v.vencido90,
                         taxa_inadimplencia = taxa))
        end
    end

    outpath = joinpath(OUT, "scr_painel_setorial.csv")
    open(outpath, "w") do io
        println(io, "trimestre,mes,cnae,n_operacoes,carteira_ativa,vencido_90,taxa_inadimplencia")
        for r in rows
            println(io, join([r.trimestre, r.mes, "\"$(r.cnae)\"", r.n_operacoes, r.carteira_ativa,
                               r.vencido_90, r.taxa_inadimplencia], ","))
        end
    end

    setores = sort(unique(r.cnae for r in rows))
    trimestres = sort(unique(r.trimestre for r in rows))
    @printf("\n%d linhas (trimestre x CNAE) exportadas em %s\n", length(rows), outpath)
    @printf("%d setores CNAE distintos, %d trimestres (%s a %s)\n",
            length(setores), length(trimestres), first(trimestres), last(trimestres))

    # Setores presentes em TODOS os trimestres (painel balanceado, necessário para matriz de dependência)
    contagem = Dict{String,Int}()
    for r in rows
        contagem[r.cnae] = get(contagem, r.cnae, 0) + 1
    end
    balanceados = sort([s for s in setores if contagem[s] == length(trimestres)])
    @printf("%d setores presentes em TODOS os %d trimestres (painel balanceado)\n",
            length(balanceados), length(trimestres))
    for s in balanceados
        println("  - ", s)
    end
end

main()
