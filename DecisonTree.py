import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

file_aluno_mat = pd.read_csv("./student-mat.csv")
file_aluno_port = pd.read_csv("./student-por.csv")

var_filter = ['age', 'sex', 'studytime', 'failures', 'freetime', 
              'goout', 'Dalc', 'Walc', 'health', 'absences', "G1", "G2", "G3"]

file_aluno_mat = file_aluno_mat[var_filter]
file_aluno_port = file_aluno_port[var_filter]


full_aluno_csv_data = pd.concat([file_aluno_mat,file_aluno_port], ignore_index=True)


full_aluno_csv_data = full_aluno_csv_data.drop_duplicates(var_filter)

full_aluno_csv_data["sex"] = full_aluno_csv_data["sex"].map({"M": 0, "F": 1})

full_aluno_csv_data['alcoholic'] = (full_aluno_csv_data['Dalc'] + full_aluno_csv_data['Walc']) > 5

x = full_aluno_csv_data.drop(columns=["alcoholic"])  
y = full_aluno_csv_data["alcoholic"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

modeloTreeObj = DecisionTreeRegressor(random_state=42, max_depth=5)
modeloTreeObj.fit(x_treino, y_treino)

previsao = modeloTreeObj.predict(x_teste)
r2 = r2_score(y_teste, previsao)

print(f"Acurácia (R²): {r2}")