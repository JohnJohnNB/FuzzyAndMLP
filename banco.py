import numpy as np
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Criação do sistema fuzzy

# 1. Definir o universo de discurso para cada variável
# Universo de entrada para Massa (M)
massa = ctrl.Antecedent(np.arange(45, 81, 1), 'massa')

# Universo de entrada para Altura (A)
altura = ctrl.Antecedent(np.arange(160, 181, 1), 'altura')

# Universo de saída para Grau de Risco (R)
grau_risco = ctrl.Consequent(np.arange(18, 36, 1), 'grau_risco')

# 2. Definir as funções de pertinência para Massa
massa['baixa'] = fuzz.trapmf(massa.universe, [45, 45, 55, 64])
massa['média baixa'] = fuzz.trimf(massa.universe, [50, 60, 68])
massa['média'] = fuzz.trimf(massa.universe, [53, 63, 72])
massa['média alta'] = fuzz.trimf(massa.universe, [56, 67, 77])
massa['alta'] = fuzz.trapmf(massa.universe, [58, 70, 80, 80])

# 3. Definir as funções de pertinência para Altura
altura['baixa'] = fuzz.trapmf(altura.universe, [157, 157, 162, 163])  # Trapézio
altura['média baixa'] = fuzz.trimf(altura.universe, [162, 165, 168])  # Triângulo
altura['média'] = fuzz.trimf(altura.universe, [167, 170, 173])        # Triângulo
altura['média alta'] = fuzz.trimf(altura.universe, [172, 175, 178])   # Triângulo
altura['alta'] = fuzz.trapmf(altura.universe, [177, 180, 183, 183])   # Trapézio

# 4. Definir as funções de pertinência para Grau de Risco
grau_risco['saudável'] = fuzz.trapmf(grau_risco.universe, [18, 18, 23, 25])
grau_risco['moderado'] = fuzz.trimf(grau_risco.universe, [24, 27, 30])
grau_risco['alto'] = fuzz.trapmf(grau_risco.universe, [29, 31, 35, 35])

# 5. Criar as regras fuzzy (baseadas na tabela fornecida)
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

# 6. Criar o sistema fuzzy
risco_ctrl = ctrl.ControlSystem(rules)
risco_simulador = ctrl.ControlSystemSimulation(risco_ctrl)

# Definir intervalos de entrada
massa_values = np.linspace(45, 80, 32)   # 32 valores para Massa
altura_values = np.linspace(157, 183, 32)  # 32 valores para Altura

# Inicializar lista para armazenar dados
data = []

# Iterar sobre todas as combinações de massa e altura
for massa_val in massa_values:
    for altura_val in altura_values:
        # Simular o sistema fuzzy para cada combinação
        risco_simulador.input['massa'] = massa_val
        risco_simulador.input['altura'] = altura_val
        risco_simulador.compute()
        
        # Obter o grau de risco (saída defuzzificada)
        grau_risco_val = risco_simulador.output['grau_risco']
        
        # Adicionar entrada e saída ao banco de dados
        data.append([massa_val, altura_val, grau_risco_val])

# Converter lista para DataFrame
df = pd.DataFrame(data, columns=['Massa', 'Altura', 'Grau de Risco'])

# Salvar banco de dados em um arquivo CSV
df.to_csv('banco_dados_fuzzy.csv', index=False)

print("Banco de dados gerado e salvo em 'banco_dados_fuzzy.csv'")
