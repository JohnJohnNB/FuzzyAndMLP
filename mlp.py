import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Carregar o banco de dados gerado
df = pd.read_csv('banco_dados_fuzzy.csv')

# 2. Dividir os dados em X (entradas) e y (saída)
X = df[['Massa', 'Altura']].values  # Entradas
y = df['Grau de Risco'].values      # Saída

# 3. Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4. Escalar os dados (opcional, mas recomendado para redes neurais)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Treinar o modelo de regressão usando MLP
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=1)
mlp.fit(X_train_scaled, y_train)

# 6. Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test_scaled)

# 7. Calcular o Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_test, y_pred)

# 8. Exibir os resultados
print("Erro Quadrático Médio (MSE):", mse)
print("R² Score (Treinamento):", mlp.score(X_train_scaled, y_train))
print("R² Score (Teste):", mlp.score(X_test_scaled, y_test))
