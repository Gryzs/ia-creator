import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

tabela = pd.read_csv("clientes.csv")
# print(tabela.info())

code = LabelEncoder()

tabela["profissao"] = code.fit_transform(tabela["profissao"])
tabela["mix_credito"] = code.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = code.fit_transform(tabela["comportamento_pagamento"])

print(tabela.info())

y = tabela["score_credito"] # QUEM EU QUERO PREVER
x = tabela.drop(columns=["score_credito", "id_cliente"]) # QUEM EU QUERO USAR PARA PREVER (Irá usar a tabela inteira exceto a score e cliente)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # test_size = percentual de teste dos dados 0.3 = 30% test -> prova, train -> treino

# Arvore de decisão
# KNN

model_forest = RandomForestClassifier()
model_knn = KNeighborsClassifier()

# Treina a IA com as "questões" x-> pergunta y-> resposta
model_forest.fit(x_train, y_train)
model_knn.fit(x_train, y_train)

# Fazer o teste = prova (x-> pergunta) ver se a IA aprendeu
prev_forest = model_forest.predict(x_test)
prev_knn = model_knn.predict(x_test)

print(accuracy_score(y_test, prev_forest)) # VERIFICAR COM O GABARITO SE A IA CONSEGUIU ACERTAR -> exibe em percentual
print(accuracy_score(y_test, prev_knn)) # VERIFICAR COM O GABARITO SE A IA CONSEGUIU ACERTAR -> exibe em percentual

# 0.8211 - forest
# 0.7376 - knn
# Assim mostrando que o metodo forest é mais eficiente 82% de acerto

new_tabela = pd.read_csv("novos_clientes.csv")
print(new_tabela.info())

new_tabela["profissao"] = code.fit_transform(new_tabela["profissao"])
new_tabela["mix_credito"] = code.fit_transform(new_tabela["mix_credito"])
new_tabela["comportamento_pagamento"] = code.fit_transform(new_tabela["comportamento_pagamento"])

prev = model_forest.predict(new_tabela)
print(prev)