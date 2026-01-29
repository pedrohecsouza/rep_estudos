
Recursos de Aprendizado: Mathematics for Machine Learning(Peter Deisenroth), CS229, MIT 18.650 Statistics for Applications, Fall 2016

A regressão logística faz parte de uma classe de algoritmos de classificação supervisionado, podendo ser indicado como o modelo mais simples de classificação.


Bases Matemáticas:

Temos várias atributos e uma resposta binária. Ou seja,  Yi|Xi  tem uma distribuição binomial ( Yi|Xi ~ B(theta,phi,b)) . A distribuição binomial faz parte das famílias exponenciais, uma família de distribuições paramétricas. A vantagem é que a função custo é sempre uma função convexa, podendo ser otimizada por métodos como Newthon Raphson, Gradient Descent Batch ou Estocástico, ou mesmo least Squares( uma vez que podemos obter uma função tal que o modelo é dado por uma relação linear) , o que torna muito fácil criar modelos para esses tipos de dados.  

Esse tipo de modelo exige que não haja interação entre as variáveis análisadas, isso é, todas as variaveis sejam estatisticamente independentes(Não chequei isso no meu modelo):

E(xy) = E(x)E(y)

Propriedades das famílias exponenciais:
$$
E(\frac{\partial log(p_\theta(y|x))}{\partial{\theta}}) = 0 
$$$$E(\frac{\partial^2log(p_\theta(y|x))}{\partial{\theta^2}}) = E(\frac{\partial log(p_\theta(y|x))}{\partial{\theta}})^2$$


DataSet Usado:

Usei o dataset de Doenças cardiacas disponível no Kaggle :

```
import kagglehub
dataset_path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
```

Os dados são distribuidos em vários fatores que podem contribuir com doenças cardiacas:

male: 0 - 1
age: 32 - 70 Anos
education: 1.0 - 4.0 Anos
currentSmoker: 0 - 1
cigsPerDay: 0.0 - 70.0 
BPMeds: 0.0 - 1.0
prevalentStroke: 0 - 1 
prevalentHyp: 0 - 1
diabetes: 0 - 1
totChol: 113.0 - 600.0 
sysBP: 83.5 - 295.0
diaBP: 48.0 - 142.5
BMI: 15.54 - 56.8 kg/m²
heartRate: 44.0 - 143.0 BPM
glucose: 40.0 - 394.0 mg/L

O objetivo é predizer o risco de 
Construí uma classe auxiliar básica para fazer analise exploratória com o seaborn.
Analise Exploratória de Dados: 

![C:\Users\dougl\Estatistica e Machine Learning\Regressão Logistica\doubleplot.png|300](file:///c%3A/Users/dougl/Estatistica%20e%20Machine%20Learning/Regress%C3%A3o%20Logistica/doubleplot.png)
Filtrei 
Tratamento de Dados:
	Os dados faltantes foram substituidos pela média dos outros
	
Fonte: Introdução à Estatística, Mario F Triola 

Base Weights:

Para tentar resolver o enviesamento dos dados, coloquei um peso no gradiente, com base na proporção dos dados:

```
def grad(self, theta):

        predictions = self.inv_Logistic_link(self.X @ theta)

        # Calcula class weights

        n_samples = len(self.Y)

        n_classes = 2

        n_class_0 = np.sum(self.Y == 0)

        n_class_1 = np.sum(self.Y == 1)

        # Weight inversamente proporcional à frequência

        weight_0 = n_samples / (n_classes * n_class_0)

        weight_1 = n_samples / (n_classes * n_class_1)

        # Aplica pesos aos erros

        weights = np.where(self.Y == 1, weight_1, weight_0)

        errors = (predictions - self.Y) * weights

        return self.X.T @ errors
```


Definição dos Parâmetros e PCA(Classificação dos Grupos)
Usando um código já nesse repositório, eu fiz o PCA para identificar quais componentes possuem maior variância. De fato, usar todos os componentes diminui a acurácia e o melhor resultado  obtido foi com 9 componentes. 
Os dados são muito desbalanceados e a distribuição não permite separar em grupos claros, o que indica que o modelo não vai ser treinado corretamente:

![C:\Users\dougl\Estatistica e Machine Learning\Regressão Logistica\PCA.png|300](file:///c%3A/Users/dougl/Estatistica%20e%20Machine%20Learning/Regress%C3%A3o%20Logistica/PCA.png)


Superfície Otimizada

A função de perda realmente é concava, e se parece com algo assim nos dois primeiros betas.

![C:\Users\dougl\Estatistica e Machine Learning\Regressão Logistica\output.png|300](file:///c%3A/Users/dougl/Estatistica%20e%20Machine%20Learning/Regress%C3%A3o%20Logistica/output.png)

Resultados:

O modelo consegui 83% de acurácia, mas olhando a matriz de confusão, percebemos que  eles tende a chutar todo mundo como negativo:

| Verdadeiro Negativo | Falso Positivo          |
| ------------------- | ----------------------- |
| 441                 | 278                     |
| **Falso Negativo**  | **Verdadeiro Positivo** |
| 65                  | 64                      |

Apesar de tentar variar o numero de componentes, a taxa de treinamento, colocar um backtracking pra reduzir a taxa de treinamento conforme a perda fosse diminuindo, não resolveu muito. Os dados são muito enviesados e é muito dificil perceber qualquer padrão nas distribuições. Mas fica o aprendizado

To do:

Adicionar outros métodos de Resolver(NR, BFGS)