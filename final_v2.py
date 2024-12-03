import numpy as np
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Sistema fuzzy

# Variáveis independentes (Antecedente):

# Massa (M)
massa = ctrl.Antecedent(np.arange(45, 81, 1), 'massa')

# Altura (A)
altura = ctrl.Antecedent(np.arange(160, 181, 1), 'altura')

# Variável dependente (Consequente):

# Grau de risco (R)
grau_risco = ctrl.Consequent(np.arange(18, 36, 1), 'grau_risco')

# Funções de pertinência para todas as variáveis:

# Para massa
massa['baixa'] = fuzz.trapmf(massa.universe, [45, 45, 55, 64])
massa['média baixa'] = fuzz.trimf(massa.universe, [50, 60, 68])
massa['média'] = fuzz.trimf(massa.universe, [53, 63, 72])
massa['média alta'] = fuzz.trimf(massa.universe, [56, 67, 77])
massa['alta'] = fuzz.trapmf(massa.universe, [58, 70, 80, 80])

# Para altura
altura['baixa'] = fuzz.trapmf(altura.universe, [157, 157, 162, 163])
altura['média baixa'] = fuzz.trimf(altura.universe, [162, 165, 168])
altura['média'] = fuzz.trimf(altura.universe, [167, 170, 173])
altura['média alta'] = fuzz.trimf(altura.universe, [172, 175, 178])
altura['alta'] = fuzz.trapmf(altura.universe, [177, 180, 183, 183])

# Para grau de risco
grau_risco['saudável'] = fuzz.trapmf(grau_risco.universe, [18, 18, 23, 25])
grau_risco['moderado'] = fuzz.trimf(grau_risco.universe, [24, 27, 30])
grau_risco['alto'] = fuzz.trapmf(grau_risco.universe, [29, 31, 35, 35])

# Visualização das funções de pertinência
massa.view()
altura.view()
grau_risco.view()

# Regras fuzzy, conforme fornecido no enunciado da prova
rules = []
rules.append(ctrl.Rule(altura['baixa'] & massa['baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['baixa'] & massa['média baixa'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['baixa'] & massa['média'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['baixa'] & massa['média alta'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['baixa'] & massa['alta'], grau_risco['alto']))

rules.append(ctrl.Rule(altura['média baixa'] & massa['baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média baixa'] & massa['média baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média baixa'] & massa['média'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['média baixa'] & massa['média alta'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['média baixa'] & massa['alta'], grau_risco['moderado']))

rules.append(ctrl.Rule(altura['média'] & massa['baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média'] & massa['média baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média'] & massa['média'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média'] & massa['média alta'], grau_risco['moderado']))
rules.append(ctrl.Rule(altura['média'] & massa['alta'], grau_risco['moderado']))

rules.append(ctrl.Rule(altura['média alta'] & massa['baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média alta'] & massa['média baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média alta'] & massa['média'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média alta'] & massa['média alta'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['média alta'] & massa['alta'], grau_risco['moderado']))

rules.append(ctrl.Rule(altura['alta'] & massa['baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['alta'] & massa['média baixa'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['alta'] & massa['média'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['alta'] & massa['média alta'], grau_risco['saudável']))
rules.append(ctrl.Rule(altura['alta'] & massa['alta'], grau_risco['saudável']))

# Sistema de controle
sistema_controle = ctrl.ControlSystem(rules)
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Banco de dados:

# Valores fictícios para as variáveis independentes
massa_values = np.linspace(45, 80, 32)
altura_values = np.linspace(157, 183, 32)

# Lista pra guardar os resultados
data = []

# Vai iterar sobre as combinações de massa e altura e fornecer uma saída
for massa_val in massa_values:
    for altura_val in altura_values:
        sistema.input['massa'] = massa_val
        sistema.input['altura'] = altura_val
        sistema.compute()

        # Pega saída, que é o grau de risco
        grau_risco_val = sistema.output['grau_risco']

        # Adiciona entradas e saída na lista, para posteriormente salvar como csv
        data.append([massa_val, altura_val, grau_risco_val])

# Converter lista para DataFrame
df = pd.DataFrame(data, columns=['Massa', 'Altura', 'Grau de Risco'])

#df = pd.read_csv('banco_dados_fuzzy.csv')

# Salva banco de dados em um arquivo CSV
df.to_csv('banco_dados_fuzzy.csv', index=False)

# Treinamento do modelo

# Divide os dados em X (entradas) e y (saída)
X = df[['Massa', 'Altura']].values
y = df['Grau de Risco'].values

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 20% para testes

#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento do modelo de regressão usando MLP
model = MLPRegressor(random_state=1, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Efetuando predições
predictions = model.predict(X_test_scaled)

# Cálculo do Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_test, predictions)

# Resultados
print("Erro Quadrático Médio (MSE):", mse)
print("R² Score (Treinamento):", model.score(X_train_scaled, y_train))
print("R² Score (Teste):", model.score(X_test_scaled, y_test))
