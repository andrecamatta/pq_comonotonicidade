# pq_comonotonicidade

Código de apoio ao artigo "Atuária para quants, parte 4: somar o Expected Shortfall por setor já é assumir a dependência mais extrema que existe", da publicação [Pílulas de Quant](https://pilulasdequant.com.br).

O repositório mede a dependência entre segmentos de risco (setores de crédito no SCR.data do Banco Central, grupos de ações da B3 por setor econômico) em janelas de calma e de estresse (recessão de 2015-2016, pandemia de 2020), e traduz essa dependência em capital agregado (Expected Shortfall e VaR a 99%) sob quatro hipóteses: independência, dependência calibrada, comonotonicidade e o pior caso de VaR via rearrangement algorithm.

## O que o código faz

`src/Comonotonicidade.jl` é o módulo numérico central. Implementa:
- Medidas de dependência: matriz de Kendall's tau par a par, e o coeficiente de comonotonicidade κ (DHAENE; VANDUFFEL; GOOVAERTS), normalizado entre 0 (independência) e 1 (comonotonicidade perfeita) a partir da razão entre a variância observada da soma e as duas variâncias extremas possíveis.
- Simulação da soma sob independência, sob comonotonicidade (via um único U~Unif(0,1) comum, aditividade de quantis) e sob uma cópula Gaussiana calibrada numa matriz de correlação (via a transformação ρ=sin(πτ/2) a partir de Kendall's tau).
- `rearrange_worst_var`: o rearrangement algorithm de Puccetti e Rüschendorf (2012) e Embrechts, Puccetti e Rüschendorf (2013), que aproxima o pior caso de VaR sob só as marginais conhecidas, discretizando a cauda de cada marginal e rearranjando iterativamente cada coluna de uma matriz N×d para minimizar a variância da soma das linhas.
- `block_bootstrap_ci`: bootstrap em bloco circular sobre observações temporais, para intervalos de confiança que preservam a autocorrelação serial.

`prepare_credit_panel.jl` lê os arquivos trimestrais do SCR.data (2012Q3-2025Q4, já baixados para a Parte 2 da série) e constrói o painel setor CNAE × trimestre de taxa de inadimplência (vencido acima de 90 dias / carteira ativa), com o mesmo recorte da Parte 2 (pessoa jurídica, modalidade Empréstimos).

`prepare_b3_panel.jl` lê os arquivos COTAHIST (2005-2025, já baixados para a Parte 1 da série) e constrói um painel de retornos diários agregados por setor econômico, usando um mapeamento ticker-setor curado manualmente (classificação B3/CVM vigente aplicada de forma retroativa, não point-in-time).

`dependence_credit.jl` e `dependence_b3.jl` medem Kendall's tau médio e o coeficiente de comonotonicidade κ por janela de calendário (série completa, calma 2017-2019, estresse 2015-2016, estresse 2020), com intervalos de confiança por bootstrap em bloco.

`aggregate_capital.jl` ajusta marginais (Gama para a taxa de inadimplência setorial, Normal para a perda de retorno setorial) na série completa de cada trilha, e traduz a dependência medida em cada janela em Expected Shortfall e VaR agregados a 99%, comparando com independência, comonotonicidade e o pior caso via rearrangement algorithm.

## Como reproduzir

Os dados brutos (SCR.data e COTAHIST) não são baixados por este repositório. Aponte as variáveis de ambiente para os repositórios das Partes 1 e 2 da série (ou copie os dados localmente):

```bash
export PANJER_RAW_DIR=/caminho/para/pq_panjer/data/raw
export BETA_DATA_DIR=/caminho/para/pq_credibilidade_beta/data/b3

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. prepare_credit_panel.jl
julia --project=. prepare_b3_panel.jl
julia --project=. dependence_credit.jl
julia --project=. dependence_b3.jl
julia --project=. aggregate_capital.jl
```

## Fonte e limitações

A trilha de crédito usa o SCR.data do Banco Central, segmento pessoa jurídica, modalidade Empréstimos, agregado por CNAE, 2012Q3-2025Q4 (54 trimestres, 19 setores presentes em todos os trimestres). A trilha de ações usa o COTAHIST da B3, 2005-2025, com uma classificação setorial curada manualmente para 59 tickers líquidos, agrupados em 10 setores.

A janela de estresse de 2020 no crédito tem só 4 observações trimestrais, insuficiente para calibrar com confiança uma cópula de 19 dimensões (por isso a tradução em capital dessa janela não foi calculada), mas suficiente para medir Kendall's tau e o coeficiente κ diretamente, com intervalo de confiança largo declarado. A trilha de ações, com amostra diária, não tem essa limitação de tamanho amostral. O rearrangement algorithm é uma heurística numérica sem prova formal de convergência, não um método com garantia matemática. Qualquer leitura numérica deste código vale para a amostra, o período e o método aqui descritos, não como propriedade geral de crises ou de mercados brasileiros.

## Referências centrais

DHAENE, J.; DENUIT, M.; GOOVAERTS, M. J.; KAAS, R.; VYNCKE, D. The concept of comonotonicity in actuarial science and finance: theory. *Insurance: Mathematics and Economics*, v. 31, n. 1, p. 3-33, 2002.

EMBRECHTS, P.; PUCCETTI, G.; RÜSCHENDORF, L. Model uncertainty and VaR aggregation. *Journal of Banking & Finance*, v. 37, n. 8, p. 2750-2764, 2013.

PUCCETTI, G.; RÜSCHENDORF, L. Computation of sharp bounds on the distribution of a function of dependent risks. *Journal of Computational and Applied Mathematics*, v. 236, n. 7, p. 1833-1840, 2012.

BERNARD, C.; RÜSCHENDORF, L.; VANDUFFEL, S. Value-at-risk bounds with variance constraints. *Journal of Risk and Insurance*, v. 84, n. 3, p. 923-959, 2017.
