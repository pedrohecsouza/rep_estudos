
# Regressão Logística

**Recursos de Aprendizado:** Mathematics for Machine Learning (Peter Deisenroth), CS229, MIT 18.650 Statistics for Applications, Fall 2016

---

## Introdução

A regressão logística faz parte de uma classe de algoritmos de **classificação supervisionada**, sendo considerado o modelo mais simples de classificação binária.

---

## Bases Matemáticas

Dado um conjunto de atributos $X$ e uma resposta binária $Y$, temos que $Y_i | X_i$ segue uma distribuição de Bernoulli:

$$Y_i | X_i \sim \text{Bernoulli}(p_i)$$

onde $p_i = \sigma(\theta^T X_i)$ e $\sigma$ é a função sigmoide (logística):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Por que usar famílias exponenciais?

A distribuição de Bernoulli pertence às **famílias exponenciais**, o que garante que a função de custo (log-likelihood negativa) seja **convexa**. Isso permite otimização eficiente por métodos como:

- Gradient Descent (Batch ou Estocástico)
- Newton-Raphson
- BFGS

### Premissa de Independência

O modelo assume que as variáveis são **estatisticamente independentes**:

$$E(XY) = E(X)E(Y)$$

> ⚠️ **Nota:** Não verifiquei essa premissa no meu modelo.

### Propriedades das Famílias Exponenciais

$$E\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta}\right) = 0$$

$$E\left(\frac{\partial^2 \log p_\theta(y|x)}{\partial \theta^2}\right) = -E\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta}\right)^2$$

> 💡 **Dica:** A segunda propriedade é conhecida como **Informação de Fisher** e é fundamental para entender a variância dos estimadores.

---

## Dataset Utilizado

Dataset de doenças cardíacas disponível no [Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression):

```python
import kagglehub
dataset_path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
```

### Variáveis do Dataset

| Variável | Descrição | Range |
|----------|-----------|-------|
| `male` | Sexo masculino | 0 - 1 |
| `age` | Idade | 32 - 70 anos |
| `education` | Escolaridade | 1.0 - 4.0 |
| `currentSmoker` | Fumante atual | 0 - 1 |
| `cigsPerDay` | Cigarros por dia | 0.0 - 70.0 |
| `BPMeds` | Medicação para pressão | 0.0 - 1.0 |
| `prevalentStroke` | Histórico de AVC | 0 - 1 |
| `prevalentHyp` | Hipertensão | 0 - 1 |
| `diabetes` | Diabetes | 0 - 1 |
| `totChol` | Colesterol total | 113.0 - 600.0 |
| `sysBP` | Pressão sistólica | 83.5 - 295.0 |
| `diaBP` | Pressão diastólica | 48.0 - 142.5 |
| `BMI` | Índice de massa corporal | 15.54 - 56.8 kg/m² |
| `heartRate` | Frequência cardíaca | 44.0 - 143.0 BPM |
| `glucose` | Glicose | 40.0 - 394.0 mg/dL |

**Objetivo:** Predizer o risco de doença cardíaca em 10 anos.

---

## Análise Exploratória de Dados

Construí uma classe auxiliar para análise exploratória com Seaborn.

![Análise Exploratória](linear_algebra_notebooks/Regressão%20Logistica/doubleplot.png)

### Tratamento de Dados

- **Dados faltantes:** substituídos pela média das demais observações.

> 📚 **Fonte:** Introdução à Estatística, Mario F. Triola

---

## Lidando com Dados Desbalanceados

### Class Weights

Para mitigar o enviesamento, apliquei pesos inversamente proporcionais à frequência de cada classe no gradiente:

```python
def grad(self, theta):
	predictions = self.inv_Logistic_link(self.X @ theta)
	
	n_samples = len(self.Y)
	n_classes = 2
	n_class_0 = np.sum(self.Y == 0)
	n_class_1 = np.sum(self.Y == 1)
	
	# Peso inversamente proporcional à frequência
	weight_0 = n_samples / (n_classes * n_class_0)
	weight_1 = n_samples / (n_classes * n_class_1)
	
	weights = np.where(self.Y == 1, weight_1, weight_0)
	errors = (predictions - self.Y) * weights
	
	return self.X.T @ errors
```

> 💡 **Conceito:** Ao dar mais peso para a classe minoritária, forçamos o modelo a "prestar mais atenção" nela durante o treinamento.

---

## Redução de Dimensionalidade com PCA

Utilizei PCA para identificar os componentes de maior variância. O melhor resultado foi obtido com **9 componentes**.

> ⚠️ Os dados são muito desbalanceados e a distribuição não permite separar grupos claros, indicando limitações no treinamento.

![PCA](linear_algebra_notebooks/Regressão%20Logistica/PCA.png)

---

## Visualização da Superfície de Custo

A função de perda é convexa (côncava no caso da log-likelihood). Visualização nos dois primeiros betas:

![Superfície de Custo](linear_algebra_notebooks/Regressão%20Logistica/output.png)

---

## Resultados

### Matriz de Confusão

|  | Predito Negativo | Predito Positivo |
|--|------------------|------------------|
| **Real Negativo** | 500 (VN) | 219 (FP) |
| **Real Positivo** | 52 (FN) | 77 (VP) |

### Métricas de Avaliação

| Métrica | Valor |
|---------|-------|
| Acurácia | 0.68 |
| Precision | 0.26 |
| Recall | 0.60 |
| F1 Score | 0.36 |

> 📖 **Entendendo as métricas:**
> - **Precision** baixa: muitos falsos positivos
> - **Recall** razoável: consegue identificar 60% dos casos positivos
> - **F1 Score** baixo: modelo desbalanceado entre precision e recall

---

## Conclusões e Aprendizados

Apesar das tentativas de ajuste (variação de componentes, taxa de aprendizado, backtracking), os resultados foram limitados devido ao forte **desbalanceamento** dos dados.

### Possíveis Melhorias

- Técnicas de oversampling (SMOTE)
- Ajuste de threshold de decisão
- Feature engineering mais elaborado

---

## To Do

- [ ] Implementar Newton-Raphson
- [ ] Implementar BFGS
- [ ] Testar SMOTE para balanceamento
- [ ] Adicionar validação cruzada
