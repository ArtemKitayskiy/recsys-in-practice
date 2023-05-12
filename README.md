### Описание данных
Данные сформированы на основе популярного датасета [Jester Online Joke Recommender System](https://goldberg.berkeley.edu/jester-data/)  

Файл train_joke_df.csv содержит:
- UID - id пользователей
- JID - id шуток
- Rating - рейтинг шутки, который проставил пользователь   
(принимает значения от -10.00 до 10.00)

Файл test_joke_df_nofactrating.csv содержит:
- UID - id пользователей
- JID - id шуток

### 1 этап - соревновании на платформе Kaggle

Реализация модели представлена в файле svd_model.ipynb.
Кроме того этот файл позволяет получить предсказания для всех пар UID и JID.

### 2 этап - доработка решения

Файл main.py упрощает процесс получения рекомендаций.  
Считываются два файла train_joke_df.csv и test_joke_df_nofactrating.csv, на выходе формируется файл output.csv, состоящий из двух стобцов: id пользователя из test_joke_df_nofactrating.csv и соответсвующий ему список [{топ рекомендация: рейтинг топ рекомендации}, [ранжированная последовательность из 10 рекомендаций]].



