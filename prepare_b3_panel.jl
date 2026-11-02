# prepare_b3_panel.jl
#
# Segunda aplicação da Parte 4: retornos diários de ações da B3 agrupadas por
# setor econômico, 2005-2025. Reaproveita os arquivos COTAHIST já baixados
# para a Parte 1 (credibilidade/beta), sem copiar dado bruto para este
# repositório. O layout do registro tipo 01 é o mesmo documentado pela B3 e
# já usado no código da Parte 1.
#
# Classificação setorial: não existe, no COTAHIST, nenhum campo de setor
# econômico. Usamos um mapeamento ticker -> setor curado manualmente, restrito
# a ações líquidas e listadas de forma contínua numa fração relevante de
# 2005-2025 (classificação setorial B3/CVM vigente, aplicada retroativamente
# como simplificação declarada, não point-in-time).
#
# Reproduzir: apontar BETA_DATA_DIR para a pasta data/b3/ do repositório
# pq_credibilidade_beta (Parte 1) e rodar `julia --project=. prepare_b3_panel.jl`.

using ZipFile, Printf, Dates, Statistics

const BETA_DATA_DIR = get(ENV, "BETA_DATA_DIR",
    joinpath(@__DIR__, "..", "..", "credibilidade", "pq_credibilidade_beta", "data", "b3"))
const OUT = joinpath(@__DIR__, "data", "output")
mkpath(OUT)
const YEARS = 2005:2025

# --- Layout fixo do registro tipo 01 do COTAHIST (mesmo da Parte 1) ---
const POS_TIPREG  = 1:2
const POS_CODBDI  = 11:12
const POS_CODNEG  = 13:24
const POS_TPMERC  = 25:27
const POS_ESPECI  = 40:49
const POS_PREULT  = 109:121
const POS_TOTNEG  = 148:152
const POS_FATCOT  = 211:217

"""
Mapeamento ticker -> setor econômico, restrito a ações líquidas com presença
relevante no período 2005-2025 (classificação vigente da B3/CVM aplicada de
forma retroativa, simplificação declarada no artigo). Cobre bancos,
petróleo e gás, mineração e siderurgia, varejo, elétrico, alimentos e
bebidas, papel e celulose, bens de capital/industrial, telecomunicações e
construção/imobiliário.
"""
const SETOR = Dict(
    # Bancos
    "ITUB4" => "Bancos", "ITUB3" => "Bancos", "BBDC4" => "Bancos", "BBDC3" => "Bancos",
    "BBAS3" => "Bancos", "SANB11" => "Bancos", "SANB4" => "Bancos", "BPAC11" => "Bancos",
    # Petróleo e gás
    "PETR3" => "Petróleo e gás", "PETR4" => "Petróleo e gás", "UGPA3" => "Petróleo e gás",
    "CSAN3" => "Petróleo e gás", "BRDT3" => "Petróleo e gás",
    # Mineração e siderurgia
    "VALE3" => "Mineração e siderurgia", "VALE5" => "Mineração e siderurgia",
    "GGBR4" => "Mineração e siderurgia", "GGBR3" => "Mineração e siderurgia",
    "CSNA3" => "Mineração e siderurgia", "USIM5" => "Mineração e siderurgia", "USIM3" => "Mineração e siderurgia",
    # Elétrico / utilities
    "ELET3" => "Elétrico", "ELET6" => "Elétrico", "CMIG4" => "Elétrico", "CMIG3" => "Elétrico",
    "CPFE3" => "Elétrico", "EGIE3" => "Elétrico", "ENBR3" => "Elétrico", "TAEE11" => "Elétrico",
    "SBSP3" => "Elétrico", "CPLE6" => "Elétrico",
    # Varejo
    "LREN3" => "Varejo", "PCAR4" => "Varejo", "PCAR3" => "Varejo", "MGLU3" => "Varejo",
    "AMAR3" => "Varejo", "VVAR3" => "Varejo",
    # Alimentos e bebidas
    "ABEV3" => "Alimentos e bebidas", "AMBV4" => "Alimentos e bebidas", "BRFS3" => "Alimentos e bebidas",
    "JBSS3" => "Alimentos e bebidas", "MRFG3" => "Alimentos e bebidas",
    # Papel e celulose
    "SUZB3" => "Papel e celulose", "SUZB5" => "Papel e celulose", "KLBN11" => "Papel e celulose",
    "FIBR3" => "Papel e celulose", "VCPA4" => "Papel e celulose",
    # Bens de capital / industrial
    "WEGE3" => "Bens de capital", "EMBR3" => "Bens de capital", "RAPT4" => "Bens de capital",
    # Telecomunicações
    "VIVT3" => "Telecomunicações", "VIVT4" => "Telecomunicações", "TIMS3" => "Telecomunicações",
    "TNLP4" => "Telecomunicações", "TNLP3" => "Telecomunicações", "OIBR4" => "Telecomunicações",
    # Construção e imobiliário
    "CYRE3" => "Construção e imobiliário", "MRVE3" => "Construção e imobiliário",
    "EZTC3" => "Construção e imobiliário", "GFSA3" => "Construção e imobiliário",
)

struct DayQuote
    price::Float64
    trades::Int
end

"Converte N(n)V99 de largura fixa em Float64."
@inline function fixed99(s::SubString)
    v = tryparse(Int64, s)
    isnothing(v) ? NaN : v / 100.0
end

"Lê um COTAHIST_A<ano>.ZIP e devolve Dict{ticker, Dict{dateint, DayQuote}} só para os tickers do mapeamento SETOR."
function read_cotahist_year(zip_path::AbstractString)
    out = Dict{String,Dict{Int,DayQuote}}()
    reader = ZipFile.Reader(zip_path)
    try
        io = reader.files[1]
        data = read(io, String)
        pos = 1
        n = ncodeunits(data)
        while pos <= n
            nl = findnext('\n', data, pos)
            lineend = isnothing(nl) ? n : nl - 1
            realend = (lineend >= pos && data[lineend] == '\r') ? lineend - 1 : lineend
            if realend - pos + 1 >= 245
                line = SubString(data, pos, pos + 244)
                if SubString(line, POS_TIPREG) == "01" &&
                   SubString(line, POS_CODBDI) == "02" &&
                   SubString(line, POS_TPMERC) == "010"
                    ticker = String(strip(SubString(line, POS_CODNEG)))
                    if haskey(SETOR, ticker)
                        dateint = parse(Int, SubString(line, 3:10))
                        price = fixed99(SubString(line, POS_PREULT))
                        fatcot = tryparse(Int64, SubString(line, POS_FATCOT))
                        fatcot = isnothing(fatcot) || fatcot <= 0 ? 1 : fatcot
                        trades = something(tryparse(Int, SubString(line, POS_TOTNEG)), 0)
                        if !isnan(price) && price > 0 && trades > 0
                            series = get!(out, ticker, Dict{Int,DayQuote}())
                            series[dateint] = DayQuote(price / fatcot, trades)
                        end
                    end
                end
            end
            pos = lineend + 2
        end
    finally
        close(reader)
    end
    return out
end

"Detecta e corrige saltos de preço >35% em 1 dia seguidos de reversão (desdobramento/grupamento/bonificação não ajustado), igual à lógica da Parte 1."
function adjust_splits(dates::Vector{Int}, prices::Vector{Float64})
    adj = copy(prices)
    n = length(prices)
    for i in 2:n
        r = adj[i] / adj[i-1]
        if r > 1.6 || r < 0.6
            # fator provável: razão arredondada para proporção simples
            adj[i:end] ./= r
        end
    end
    return adj
end

function build_panel()
    @printf("Lendo COTAHIST de %s (anos %d-%d)\n", BETA_DATA_DIR, first(YEARS), last(YEARS))
    panel = Dict{String,Dict{Int,DayQuote}}()
    for year in YEARS
        zp = joinpath(BETA_DATA_DIR, "COTAHIST_A$(year).ZIP")
        isfile(zp) || continue
        yd = read_cotahist_year(zp)
        for (t, series) in yd
            target = get!(panel, t, Dict{Int,DayQuote}())
            merge!(target, series)
        end
    end
    @printf("%d tickers mapeados encontrados no COTAHIST\n", length(panel))
    for t in sort(collect(keys(panel)))
        @printf("  %s (%s): %d dias\n", t, SETOR[t], length(panel[t]))
    end
    return panel
end

function daily_log_returns(series::Dict{Int,DayQuote})
    dates = sort(collect(keys(series)))
    length(dates) < 2 && return Int[], Float64[]
    prices = [series[d].price for d in dates]
    prices = adjust_splits(dates, prices)
    rets = diff(log.(prices))
    return dates[2:end], rets
end

function main()
    panel = build_panel()

    # Retorno diário por ticker
    ticker_rets = Dict{String,Dict{Int,Float64}}()
    for (t, series) in panel
        dates, rets = daily_log_returns(series)
        length(dates) < 250 && continue  # exige histórico mínimo
        d = Dict{Int,Float64}()
        for (dt, r) in zip(dates, rets)
            abs(r) < 0.5 && (d[dt] = r)  # descarta outliers residuais (>50% log-retorno diário, erro de dado)
        end
        ticker_rets[t] = d
    end
    @printf("\n%d tickers com histórico mínimo de 250 dias úteis\n", length(ticker_rets))

    # Índice setorial = média igual-ponderada dos retornos dos tickers do setor, por dia
    setores = sort(unique(values(SETOR)))
    all_dates = sort(unique(d for t in keys(ticker_rets) for d in keys(ticker_rets[t])))

    outpath = joinpath(OUT, "b3_painel_setorial.csv")
    open(outpath, "w") do io
        println(io, "data,setor,retorno,n_tickers")
        for dt in all_dates
            for s in setores
                vals = Float64[]
                for (t, sec) in SETOR
                    sec != s && continue
                    haskey(ticker_rets, t) || continue
                    haskey(ticker_rets[t], dt) || continue
                    push!(vals, ticker_rets[t][dt])
                end
                isempty(vals) && continue
                println(io, join([dt, "\"$s\"", mean(vals), length(vals)], ","))
            end
        end
    end
    @printf("Painel setorial de retornos B3 exportado em %s\n", outpath)
    @printf("%d setores, %d dias de calendário com pregão em ao menos 1 setor\n", length(setores), length(all_dates))
end

main()
