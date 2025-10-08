# Relatório — parte 1: Tratamento de imagem


## 1 - A

Binarização converte níveis de cinza em dois valores (ex.: 0 e 255). Técnicas comuns:

- Threshold global (manual): usa um limiar fixo; funciona bem com iluminação uniforme, falha com sombras ou variação local.
- Otsu: escolhe o limiar automaticamente; bom quando o histograma é bimodal (dois picos bem separados).
- Adaptativa (mean/gaussian): calcula limiais locais; indicada para iluminação não uniforme.

O histograma indica a probabilidade de sucesso:

- Bimodal → boa chance de sucesso com threshold global ou Otsu.
- Distribuição contínua/sobreposta → limiar global/Otsu tendem a falhar; usar binarização adaptativa ou pré-processamento (equalização, suavização).

Em resumo: o histograma ajuda a prever se um limiar global será suficiente (picos separados) ou se métodos locais/preprocessamento serão necessários.

## 1 - B

Contraste e brilho alteram a aparência e o histograma da imagem de formas previsíveis:

- Operações (exemplos): brilho (adição/subtração de um valor constante), contraste linear (escalar os tons em torno de um ponto central), correção de gama (transformação não linear que afeta principalmente os tons médios), equalização/CLAHE.
- Efeito no histograma: brilho desloca a posição (direita/esquerda) sem mudar a forma — salvo clipping; operações de contraste, correção de gama e equalização alteram a forma (alargam, comprimem ou redistribuem os valores).
- Correlação visual ↔ estatística: média ↔ brilho; desvio padrão ↔ contraste percebido; assimetria ↔ predominância de tons claros ou escuros. Medir média, desvio padrão e inspecionar o histograma ajuda a justificar ajustes.
- Quando os ajustes pioram a qualidade: clipping (saturação nos extremos) — perda de detalhe; aumento excessivo de contraste — amplifica ruído e gera halos; equalização global em cenas com iluminação desigual — aparência artificial; prefira CLAHE quando necessário.

Recomendação prática: testar combinações simples de escala e deslocamento de intensidade, visualizar o histograma e checar média, desvio padrão e clipping antes de aplicar ajustes definitivos.

## 2 - A

- h1: Box 3×3 (média, passa‑baixa). h2: Laplaciano 3×3 (realce/alta‑frequência).
- Teoria (LTI): convolução é associativa e comutativa, logo h1*(h2I) = (h1h2)I = h2(h1*I) — todos equivalentes em ideal LTI.
- Efeito nos histogramas: Box tende a concentrar/alisar (reduz variância); Laplaciano realça bordas e amplia caudas (valores positivos/negativos).
- Por que pode haver diferenças na prática: tratamento de bordas, normalização/escala, tipos numéricos e arredondamento ou qualquer passo não‑linear entre filtros.
- Conclusão: teoricamente iguais; na prática, padronize padding, tipos e normalização para obter equivalência numérica.

## 3

1. Quais filtros espaciais têm efeitos melhores e por quê?

- Bilateral: melhor compromisso entre redução de ruído e preservação de arestas, pois condiciona a suavização ao contraste local.
- Mediana: superior para ruído impulsivo (salt‑and‑pepper), pois substitui outliers sem borrar contornos.
- Gaussiano (espacial): indicado para suavização homogênea; reduz altas frequências de forma gradual sem introduzir blocos.
- Filtros derivativos (Sobel, Laplaciano): adequados para realce de bordas, normalmente usados após suavização para evitar amplificação de ruído.

2. E quanto aos filtros no domínio da frequência, quais são os
   melhores parâmetros que geram melhores resultados?

- Preferir filtros Gaussiano ou Butterworth; evitar filtro ideal (retangular) por causar ringing.
- Parâmetros práticos: cutoff entre 0.1 e 0.3 da frequência de Nyquist; Butterworth com ordem 1–2 para transição suave. Ajuste fino conforme a textura e o nível de ruído da imagem.

3. Comparação entre abordagens passa‑baixa (espacial vs frequência)

- Preservação de contornos: filtros espaciais adaptativos (bilateral, mediana) tendem a obter melhores resultados porque consideram informação local de contraste.
- Artefatos: Gaussiano espacial e Gaussiano em frequência apresentam comportamento similar; filtros ideais em frequência produzem ringing; Box espacial gera artefatos visíveis.
- Eficiência: para kernels grandes, implementação via FFT (domínio da frequência) é mais eficiente; para kernels pequenos/separáveis, convolução espacial é preferível.

## Parte 2 — Segmentação

### 1.b Pré‑processamento e reavaliação do Canny

Aplicou‑se um pré‑processamento único antes de rodar o Canny com os mesmos parâmetros. Recomenda‑se desfoque Gaussiano leve (3×3, σ ≈ 1): reduz detecções espúrias causadas por ruído sem apagar as arestas principais. Para ruído impulsivo use mediana 3×3; para máxima preservação de contornos, bilateral (mais custoso).

Avaliação: comparar contagem de arestas, SSIM entre mapas de borda e inspecionar histogramas de gradiente. Em prática, Gaussiano 3×3 (σ≈1) mostrou bom compromisso entre precisão e simplicidade.

### 1.c Aplicação nos grupos 2 — resultados e correções

Aplicando o mesmo pré‑processamento e parâmetros de Canny ao grupo 2, os resultados foram satisfatórios quando ruído e contraste eram similares; caso contrário, verificou‑se perda de arestas fracas ou aumento de falsos positivos. Correções simples: ajustar os limiares do Canny e, se necessário, adaptar o pré‑processamento (aumentar σ do Gaussiano, usar mediana ou CLAHE). Avaliar por contagem de arestas e SSIM; reportar médias e desvios.

### 2.a Mean Shift (aplicação e análise)

Procedimento resumido: testar o Mean Shift inicialmente em versões reduzidas das imagens para economizar tempo; realizar varredura simples em hs (raio espacial) e hr (raio de cor) para encontrar compromisso entre redução de ruído e preservação de regiões.

Faixas práticas: hs ≈ 4–20 (aumentar hs funde regiões próximas espacialmente), hr ≈ 10–50 (aumentar hr funde regiões por similaridade de cor). Começar em hs=8, hr=16 e ajustar conforme resultado visual.

O que observar: nº de regiões (over/under‑segmentation), preservação de contornos, clareza das bordas após aplicação de Canny, e custo computacional. Escolha final deve equilibrar segmentação estável e preservação de detalhe.
