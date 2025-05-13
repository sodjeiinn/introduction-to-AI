import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 
# на группы
from sklearn.metrics import classification_report, accuracy_score

# Загружаем CSV
df = pd.read_csv("Train.csv")


df = df.drop("ID", axis=1)

# Заполняем пропуски если есть пустые ячейки не будет раб 
# середина 
df['Ever_Married'].fillna('Unknown', inplace=True)
df['Graduated'].fillna('Unknown', inplace=True)
df['Profession'].fillna('Unknown', inplace=True)
df['Work_Experience'].fillna(df['Work_Experience'].median(), inplace=True)
df['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)
df['Var_1'].fillna('Unknown', inplace=True)

# мод не понимает слова ток числа 
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Разделение X   это признаки все чт влияет на рез, у predict это куда поп человек
X = df.drop("Segmentation", axis=1)
y = df["Segmentation"]

# Разделение на обучающую и тестовую 80 обуч и 20 как работ 42 чтобы кажд раз делилось одинак
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train) 


# x analiz y predict сохранить и новыми данными 
# х анализирует у учится угадывать 

# Предсказание и оценка
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# кр более подробная оценка