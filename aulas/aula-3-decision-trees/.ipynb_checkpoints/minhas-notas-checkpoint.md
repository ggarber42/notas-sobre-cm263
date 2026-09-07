# Anotações sobre árvores de decisão

É um modelo supervisionado utilizado para classificação (decisão), utiliza testes encadeados e é de fácil interpretabilidade (é fácil rastrear as decisões).

## Heurísticas de seleção de atributos

As árvores de decisão se baseiam na estratégia de **dividir e conquistar**, então a cada subdivisão a ideia é gerar um nó puro com uma só carcterística. Então, estabelecendo uma medida de pureza, podemos selecionar qual atributo selecionar para se aproximar mais desse cenário.

### Algoritmos para construção de árvores

Gini Index - Algoritmo CART

Ganho de informação - Algoritmo ID3

Razão de ganho - Algoritmo C4.5

#### Índice de Gini - Grau de Impureza

Medida de desigualdade, **0** é **totalmente igual** e **1 totalmente desigual**.


$$\text{Gini Impurity} = 1 - \sum_{i=1}^{K} p_i^2$$

pi é a probabilidade de uma instância pertencer à classe i, estimada a partir de D; m é o número de classes.

**Exemplo:**

A base de dados possui **14 instâncias** no total:

**Classe Alvo ($buys\_computer$):** $yes$ (9 instâncias) e $no$ (5 instâncias).


| RID | age | income | student | credit\_rating | buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | youth | high | no | fair | no |
| 2 | youth | high | no | excellent | no |
| 3 | middle\_aged | high | no | fair | yes |
| 4 | senior | medium | no | fair | yes |
| 5 | senior | low | yes | fair | yes |
| 6 | senior | low | yes | excellent | no |
| 7 | middle\_aged | low | yes | excellent | yes |
| 8 | youth | medium | no | fair | no |
| 9 | youth | low | yes | fair | yes |
| 10 | senior | medium | yes | fair | yes |
| 11 | youth | medium | yes | excellent | yes |
| 12 | middle\_aged | medium | no | excellent | yes |
| 13 | middle\_aged | high | yes | fair | yes |
| 14 | senior | medium | no | excellent | no |

---

Primeiro partimos da feature alvo, `buys_computer`

$$\text{Gini}(D) = 1 - \left(\frac{9}{14}\right)^2 - \left(\frac{5}{14}\right)^2 \approx 1 - 0.413 - 0.128 = 0.459$$



No caso a _feature_ **income** é categórica e com 3 possíveis valores (low, medium, high), por isso a gente tem que agrupar elas em 2 grupos porque o algoritmo **CART** é busca divisões binárias.

Então fazemos as seguintes divisões:

 **$\text{income} \in \{\text{low, medium}\}$**

 **$\text{income} \in \{\text{high, medium}\}$**

 **$\text{income} \in \{\text{high, low}\}$**

 E pelo princípio do conjunto complementar temos:

 **$\text{income} \in \{\text{low, medium}\} = \text{income} \in \{\text{high}\}$**

  **$\text{income} \in \{\text{high, medium}\} = \text{income} \in \{\text{low}\}$**

   **$\text{income} \in \{\text{high, low}\} = \text{income} \in \{\text{medium}\}$**

E selecionamos aquele cujo $\Delta \text{Gini}$ for o menor (em relação a ao Gini da feature alvo)

Cálculo para **$\text{Gini}_{\text{income} \in \{\text{low, medium}\}}$**:

$$\text{Gini}_{\text{income} \in \{\text{low, medium}\}}(D) = \frac{|D_1|}{|D|} \text{Gini}(D_1) + \frac{|D_2|}{|D|} \text{Gini}(D_2)$$

De 10 low e medium's temos 7 com _yes_ e 3 com _no_ na coluna alvo.

$$\text{Gini}(D_1) = 1 - \left(\frac{7}{10}\right)^2 - \left(\frac{3}{10}\right)^2 = 1 - 0.49 - 0.09 = 0.42$$

No high temos 2 _yes_ e 2 _no_:

$$\text{Gini}(D_2) = 1 - \left(\frac{2}{4}\right)^2 - \left(\frac{2}{4}\right)^2 = 1 - 0.25 - 0.25 = 0.50$$

E low e medium temos 10 no total e high temos 4:

$$\text{Gini}_{\text{income} \in \{\text{low, medium}\}}(D) = \frac{10}{14}(0.42) + \frac{4}{14}(0.50) \approx 0.443$$

Fazendo isso para os outros dois conjuntos:

*   $\text{Gini}_{\text{income} \in \{\text{low, medium}\}}(D) = \text{Gini}_{\text{income} \in \{\text{high}\}}(D) = 0.443$
  

*   $\text{Gini}_{\text{income} \in \{\text{low, high}\}}(D) = \text{Gini}_{\text{income} \in \{\text{medium}\}}(D) = 0.458$
  

*   $\text{Gini}_{\text{income} \in \{\text{medium, high}\}}(D) = \text{Gini}_{\text{income} \in \{\text{low}\}}(D) = 0.450$

-----

* $\Delta \text{Gini}_1 = 0.459 - 0.443 = 0.016$
* $\Delta \text{Gini}_2 = 0.459 - 0.458 = 0.001$
* $\Delta \text{Gini}_3 = 0.459 - 0.450 = 0.009$

Escolhemos **o menor Gini** que é o $\text{Gini}_{\text{income} \in \{\text{high}\}}$ porque ele resulta no maior (maximização) **$\Delta \text{Gini}$** (maior redução do grau de impureza).


## Otimização de Hiperparâmetros (pré e pós poda)


* Pré-poda `max_depth` - limita a profundidade da árvore
* Pós-poda `ccp_alphas` - custo complexidade 

#### Viés vs Variância (bias vs variance) trade-off

<u>**Viés**</u>

Alto viés implica em **árvores rasas** que não conseguem capturar os padrões relevantes dos dados. (underfitting)

<u>**Variância**</u>

Árvores profundas tendem a ter alta variância e acabam capturando os ruídos dos dados de treinamento.

#### Exemplo:

Primeiro importa-se as libs e os dados

```python
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target
```

#### <u>Estratégia holdout (80/20) - sem otimizar os hiperparâmetros</u>

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_random_state)

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)

y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

acuracia_treino = accuracy_score(y_train, y_pred_train)
acuracia_teste = accuracy_score(y_test, y_pred_test)

print(f"Acurácia de Treino: {acuracia_treino:.4f}")
print(f"Acurácia de Teste (Generalização): {acuracia_teste:.4f}")
```

`Acurácia de Treino: 1.0000`

`Acurácia de Teste (Generalização): 0.9386`


#### <u>Estratégia holdout (70/15/15) - otimizando os hiperparâmetros</u>

Primeiro separam-se os dados para testes

```python
test_size = 0.15
val_size_total = 0.15 

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)
```

E depois para treino e validação:

```python
val_size_adjusted = val_size_total / (1 - test_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
)
```

**Pré-póda** - `max_depth`

```python
depths = range(1, 21)
train_scores = []
val_scores = []

for d in depths:
   dt = DecisionTreeClassifier(max_depth=d, random_state=42)
   dt.fit(X_train, y_train)
   train_score = accuracy_score(y_train, dt.predict(X_train))
   val_score = accuracy_score(y_val, dt.predict(X_val))

   train_scores.append(train_score)
   val_scores.append(val_score)

```

Plotando

```python
plt.plot(depths, train_scores, label='Treino')
plt.plot(depths, val_scores, label='Validação')
plt.xlabel('max_depth')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
```


![alt text](image.png)

Quanto maior a profundidade maior a variância (overfitting)


**Pos-póda** - `ccp_aplha`

Definindo os alphas

```python
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

print(alphas)
```

`[0.         0.00238632 0.0024966  0.00327488 0.00429537 0.00447803
 0.00484871 0.0049958  0.0065135  0.00944584 0.01406738 0.01436818
 0.0595357  0.32050608]`

 

