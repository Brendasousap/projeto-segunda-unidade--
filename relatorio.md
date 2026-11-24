# Relatório — Segunda Lista de Projetos: Sistemas Inteligentes

## Problema 1 — Adaline e Ionosphere

O dataset Ionosphere contém atributos numéricos que descrevem retornos de radar. Os rótulos são convertidos para +1 (`good`) e −1 (`bad`). O script verifica valores ausentes e atributos constantes, divide os dados estratificadamente e ajusta o `StandardScaler` somente no conjunto de treino.

### Por que padronizar?

A padronização é \(z_j=(x_j-\mu_j)/\sigma_j\). Sem ela, atributos com escalas grandes podem dominar o gradiente. Para a perda quadrática, a Hessiana da função de custo é proporcional a \(X^TX/N\); a velocidade de convergência da descida do gradiente depende do maior autovalor e do condicionamento dessa matriz. Escalas muito diferentes aumentam o número de condição e podem exigir taxas de aprendizagem muito pequenas para evitar oscilações/divergência.

### Métricas e interpretação

Execute `problema1_adaline.py`. O programa gera `p1_curvas_adaline.png`, matriz de confusão, acurácia, precisão, recall e F1-score. Escolha a taxa de aprendizagem que apresente curva estável e menor perda final; se uma taxa divergir, descarte-a mesmo que a taxa seja alta.

Na aplicação descrita, um falso positivo da classe `bad` — prever `good` quando o retorno é `bad` — pode ocultar a ausência de estrutura coerente. Em monitoramento, esse erro pode levar o sistema a considerar confiável um retorno que não é confiável. A gravidade depende do uso operacional, mas deixar passar a classe problemática costuma ser um risco importante.

## Problema 2 — MLP e NSL-KDD

O script lê `KDDTrain+.txt` e `KDDTest+.txt`, descarta `difficulty_level`, converte `normal` em 0 e qualquer ataque em 1, aplica one-hot encoding a `protocol_type`, `service` e `flag`, e padroniza os atributos numéricos usando apenas estatísticas do treino. A MLP contém camadas ocultas de 128 e 64 neurônios com ReLU e saída linear de um neurônio. `BCEWithLogitsLoss` combina numericamente a sigmoide e a entropia cruzada binária.

A validação é formada pelos últimos 20% do arquivo de treino, sem embaralhar a divisão. O DataLoader embaralha os mini-lotes do subconjunto de treino. A parada antecipada conserva os pesos com menor loss de validação.

### Métricas e decisão

Execute `problema2_mlp.py` com os dois arquivos do NSL-KDD. O limiar 0,5 transforma probabilidades em rótulos. Um falso negativo (ataque previsto como normal) tende a ser mais crítico para segurança, pois permite a passagem de uma intrusão. Reduzir o limiar normalmente aumenta o recall de ataques e reduz falsos negativos, mas pode aumentar falsos positivos e alertas desnecessários. A escolha deve refletir o custo operacional dos dois tipos de erro.

## Problema 3 — CNN e EuroSAT

O script baixa o EuroSAT RGB, apresenta a contagem por classe e uma grade com cinco imagens por classe. A divisão é estratificada em 70% treino, 15% validação e 15% teste. A média e o desvio-padrão por canal são calculados somente no treino. A CNN tem convoluções 3→32→64→128, kernel 3×3, padding 1, ReLU e max pooling 2×2 após cada convolução. Após três pools, a dimensão espacial passa de 64×64 para 8×8; portanto, a camada densa recebe 128×8×8 atributos e produz 128 neurônios, seguidos por 10 logits. A entropia cruzada do PyTorch recebe logits, sem softmax explícita.

O treinamento roda até 30 épocas com parada antecipada baseada na loss de validação. São geradas curvas de loss e acurácia, matriz de confusão e métricas por classe. O mesmo processo é repetido com espelhamentos horizontal/vertical e rotação apenas no treino.

### Análise dos resultados

Use as curvas e os relatórios impressos pelo script para preencher a comparação. Há indício de overfitting quando a loss de treino continua caindo enquanto a loss de validação para de cair ou sobe, ou quando a acurácia de treino cresce e a de validação estagna. Identifique os pares mais confundidos observando os maiores valores fora da diagonal da matriz de confusão; examine exemplos dessas imagens antes de explicar as semelhanças visuais.

Espelhamentos e rotações são adequados quando a orientação da imagem não altera a classe de uso/cobertura do solo. Eles aumentam a variedade do treino e podem reduzir a memorização. Não devem ser aplicados à validação/teste porque esses conjuntos devem representar dados de avaliação estáveis e comparáveis, sem transformações aleatórias que alterem a medição.

## Reprodutibilidade e limitações

Os scripts fixam sementes aleatórias, mas os resultados exatos podem variar conforme versões de bibliotecas, hardware e disponibilidade de GPU. As métricas finais devem ser preenchidas após execução. Não se devem inventar números: registre as saídas reais e inclua as figuras geradas no relatório final.
