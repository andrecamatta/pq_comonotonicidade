# pq_comonotonicidade

Código de apoio ao artigo "Atuária para quants, parte 4: somar o Expected Shortfall por setor pressupõe a dependência mais extrema compatível com cada risco", da publicação [Pílulas de Quant](https://pilulasdequant.com.br).

O repositório mede a dependência entre segmentos de risco (setores de crédito no SCR.data do Banco Central, grupos de ações da B3 por setor econômico) em janelas de calma e de estresse (recessão de 2015-2016, pandemia de 2020), e traduz essa dependência em capital agregado (Expected Shortfall e VaR a 99%) sob quatro hipóteses: independência, dependência calibrada, comonotonicidade e o pior caso de VaR via rearrangement algorithm.

## O que o código faz

`src/Comonotonicidade.jl` é o módulo numérico central. Implementa:
- Medidas de dependência: matriz de Kendall's tau par a par, e a medida de dependência ρc de Dhaene, Linders, Schoutens e Vyncke (2014), ρc = (Var(S) - ΣVar(Xᵢ)) / (Var(Sᶜ) - ΣVar(Xᵢ)). Var(Sᶜ) é calculada a partir de `comonotonic_rearrangement`, que ordena cada coluna de forma ascendente e soma linha a linha, não pela aproximação (Σσᵢ)² (que só é exata quando a correlação par a par sob comonotonicidade é exatamente 1, caso de marginais ligadas por transformação linear). ρc=1 corresponde ao rearranjo já estar na configuração comonotônica (dependência máxima compatível com as marginais observadas); ρc=0, a covariância par a par nula em média, não a independência (covariância nula não implica independência fora do caso Gaussiano); ρc pode ser negativo.
- `first_differences`: primeiras diferenças por coluna, usadas para estacionarizar a taxa de inadimplência setorial antes de medir dependência (ver "Fonte e limitações").
- `empirical_tail_dependence` e `mean_tail_dependence`: dependência de cauda empírica (λ superior e inferior) por contagem de postos. `gaussian_copula_tail_dependence`: a mesma medida, simulada sob uma cópula Gaussiana calibrada num dado Kendall's tau, para comparar com o valor empírico.
- Simulação da soma sob independência, sob comonotonicidade (via um único U~Unif(0,1) comum, aditividade de quantis) e sob uma cópula Gaussiana calibrada numa matriz de correlação (via a transformação ρ=sin(πτ/2) a partir de Kendall's tau, corrigida para positiva definida pela projeção de Higham (2002) para a matriz de correlação mais próxima, `nearest_corr_higham`, em `aggregate_capital.jl`; a versão anterior usava corte de autovalores com renormalização, ainda disponível em `nearest_psd`). `diagnostico_psd.jl` reporta a positividade da matriz sin(πτ/2) por janela, a alteração causada por cada correção e a comparação com equicorrelação no τ médio (resultados em `data/output/psd_diagnostico.csv` e `*_capital_equicorrelacao.csv`).
- `rearrange_worst_var`: o rearrangement algorithm de Puccetti e Rüschendorf (2012) e Embrechts, Puccetti e Rüschendorf (2013), que aproxima o pior caso de VaR sob só as marginais conhecidas, discretizando a cauda de cada marginal e rearranjando iterativamente cada coluna de uma matriz N×d para minimizar a variância da soma das linhas. Por teoria (ARTZNER et al., 1999; EMBRECHTS; PUCCETTI; RÜSCHENDORF, 2013), VaR(S) ≤ ES(S) ≤ Σ ES marginal para qualquer soma S: o pior VaR encontrado nunca deve superar esse teto.
- `block_bootstrap_ci`: bootstrap em bloco circular sobre observações temporais, para intervalos de confiança que preservam a autocorrelação serial.

`test/runtests.jl` testa o módulo em casos com resultado conhecido: ρc=1 em construção comonotônica exata, ρc≈0 sob independência, ρc fortemente negativo sob contramonotonicidade exata (d=2, não necessariamente -1 exato com a fórmula de Dhaene et al., ver comentário no teste), aditividade de quantis da soma comonotônica contra marginais conhecidas, a relação VaR(S) ≤ ES(S) ≤ Σ ES marginal para o pior caso via rearrangement algorithm, primeiras diferenças, dependência de cauda em dados comonotônicos por construção, o rearranjo comonotônico em si, e a cauda da cópula Gaussiana sob independência (τ=0).

`prepare_credit_panel.jl` lê os arquivos trimestrais do SCR.data (2012Q3-2025Q4, já baixados para a Parte 2 da série) e constrói o painel setor CNAE × trimestre de taxa de inadimplência (vencido acima de 90 dias / carteira ativa), com o mesmo recorte da Parte 2 (pessoa jurídica, modalidade Empréstimos).

`prepare_b3_panel.jl` lê os arquivos COTAHIST (2005-2025, já baixados para a Parte 1 da série) e constrói um painel de retornos diários agregados por setor econômico, usando um mapeamento ticker-setor curado manualmente (classificação B3/CVM vigente aplicada de forma retroativa, não point-in-time).

`dependence_credit.jl` mede Kendall's tau médio e ρc por janela de calendário (série completa, calma 2017-2019, estresse 2015-2016, estresse 2020) sobre **primeiras diferenças** da taxa de inadimplência, não sobre o nível (ver "Fonte e limitações"), com intervalos de confiança por bootstrap em bloco quando a janela tem 8 observações ou mais, e dependência de cauda empírica (e sua versão via cópula Gaussiana) na série completa.

`dependence_b3.jl` faz o mesmo para a trilha de ações, sobre o retorno diário (já estacionário por construção, sem necessidade de diferenciação adicional).

`aggregate_capital.jl` ajusta marginais no nível (Gama para a taxa de inadimplência setorial, t de Student com 5 graus de liberdade para a perda de retorno setorial, cauda mais pesada que a Normal) na série completa de cada trilha, calibra a estrutura de dependência de cada janela sobre a série estacionária, e traduz em Expected Shortfall e VaR agregados a 99%, comparando com independência, comonotonicidade e o pior caso via rearrangement algorithm. Para o crédito, a tradução em capital só é calculada quando a janela tem 12 observações estacionárias ou mais (calibrar uma cópula de 19 dimensões com menos que isso não é defensável), o que exclui as duas janelas de estresse dessa trilha, medidas só via `dependence_credit.jl`.

## Como reproduzir

Os dados brutos (SCR.data e COTAHIST) não são baixados por este repositório. Aponte as variáveis de ambiente para os repositórios das Partes 1 e 2 da série (ou copie os dados localmente):

```bash
export PANJER_RAW_DIR=/caminho/para/pq_panjer/data/raw
export BETA_DATA_DIR=/caminho/para/pq_credibilidade_beta/data/b3

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. prepare_credit_panel.jl
julia --project=. prepare_b3_panel.jl
julia --project=. dependence_credit.jl
julia --project=. dependence_b3.jl
julia --project=. aggregate_capital.jl
```

Os CSVs de resultado final (`data/output/*_dependencia_*.csv`, `data/output/*_capital_agregado.csv`) estão versionados neste repositório. Os painéis intermediários construídos a partir do dado bruto (`scr_painel_setorial.csv`, `b3_painel_setorial.csv`) não estão, por tamanho e por serem reconstruíveis a partir dos scripts acima.

## Fonte e limitações

A trilha de crédito usa o SCR.data do Banco Central, segmento pessoa jurídica, modalidade Empréstimos, agregado por CNAE, 2012Q3-2025Q4 (54 trimestres, 19 setores presentes em todos os trimestres). A trilha de ações usa o COTAHIST da B3, 2005-2025, com uma classificação setorial curada manualmente para 59 tickers líquidos, agrupados em 10 setores.

A taxa de inadimplência setorial tem nível com tendência comum entre setores (por exemplo, queda conjunta em 2020), o que infla qualquer medida de dependência calculada sobre o nível da série. Por isso, a dependência entre setores de crédito é medida sobre primeiras diferenças (`first_differences`), não sobre o nível. O retorno diário de ações já é, por definição, a variação período a período do preço, e não precisa dessa transformação.

A janela de estresse de 2020 no crédito tem só 4 observações em diferenças, abaixo do mínimo de 8 adotado para que o bootstrap em bloco produza um intervalo de confiança minimamente informativo (com menos observações, o número de configurações distintas de reamostragem fica pequeno demais). O ponto estimado de Kendall's tau e de ρc é reportado para essa janela, sem intervalo de confiança. A janela de 2015-2016 (8 observações, 2 blocos) tem intervalo indicativo, não uma inferência com a mesma força das demais janelas. A tradução em capital exige ainda mais observações (12, nesta trilha de 19 dimensões), o que exclui tanto a janela de 2020 quanto a de 2015-2016 da tradução em capital, calculadas só via medidas de dependência.

A unidade de capital da trilha de crédito é a soma não ponderada das taxas de inadimplência setoriais, não uma soma de saldo em reais nem uma métrica ponderada por exposição de carteira. A marginal Gama de cada setor de crédito é ajustada sobre o nível da taxa de inadimplência ao longo de 54 trimestres com tendência e quebras de regime, não sobre uma série estacionária. A cópula Gaussiana usada para a dependência calibrada captura a correlação de postos medida por Kendall's tau; comparada com a dependência de cauda empírica (via `gaussian_copula_tail_dependence`), essa aproximação se mostrou razoável nesta amostra, com só um pequeno excesso na cauda inferior das ações. A calibração da cópula da janela calma de crédito mistura escalas (correlação em diferenças, marginais em nível) e calibra 171 correlações par a par com só 12 observações, uma limitação declarada no artigo. O rearrangement algorithm é uma heurística numérica sem prova formal de convergência, não um método com garantia matemática. Não existe nenhum indicador público do Banco Central de correlação de inadimplência entre setores contra o qual validar a matriz estimada aqui.

Qualquer leitura numérica deste código vale para a amostra, o período e o método aqui descritos, não como propriedade geral de crises ou de mercados brasileiros.

## Referências centrais

ARTZNER, P.; DELBAEN, F.; EBER, J.-M.; HEATH, D. Coherent measures of risk. *Mathematical Finance*, v. 9, n. 3, p. 203-228, 1999.

DHAENE, J.; DENUIT, M.; GOOVAERTS, M. J.; KAAS, R.; VYNCKE, D. The concept of comonotonicity in actuarial science and finance: theory. *Insurance: Mathematics and Economics*, v. 31, n. 1, p. 3-33, 2002.

DHAENE, J.; LINDERS, D.; SCHOUTENS, W.; VYNCKE, D. A multivariate dependence measure for aggregating risks. *Journal of Computational and Applied Mathematics*, v. 263, p. 78-87, 2014.

EMBRECHTS, P.; PUCCETTI, G.; RÜSCHENDORF, L. Model uncertainty and VaR aggregation. *Journal of Banking & Finance*, v. 37, n. 8, p. 2750-2764, 2013.

PUCCETTI, G.; RÜSCHENDORF, L. Computation of sharp bounds on the distribution of a function of dependent risks. *Journal of Computational and Applied Mathematics*, v. 236, n. 7, p. 1833-1840, 2012.
