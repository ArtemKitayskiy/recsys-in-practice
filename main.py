import numpy as np
import pandas as pd
from surprise import dump

def get_top_jokes(df, k):
    # сортировка по рейтингу
    sorted_df = df.sort_values(by=['UID', 'Rating'], ascending=[True, False])
    grouped_df = sorted_df.groupby('UID')

    # создание словаря для каждого пользователя
    user_jokes = {}
    for index, row in sorted_df.iterrows():
        uid = int(row['UID'])
        jid = int(row['JID'])

        if uid not in user_jokes:
            top_joke_dict = {jid: row['Rating']}
            user_jokes[uid] = [top_joke_dict, [jid]]

        elif len(user_jokes[uid][1]) < k:
            user_jokes[uid][1].append(jid)

    # создание списка для всех пользователей
    result = []
    for uid, jokes in user_jokes.items():
        result.append([uid, jokes])
    
    return result, user_jokes

# загрузка модели
_, svd = dump.load('svd_model_optuna')

train = pd.read_csv('train_joke_df.csv')
test = pd.read_csv('test_joke_df_nofactrating.csv', index_col=0)
test.index.name = None

# создаем список всех пар пользователь-шутка
users = range(1, len(train['UID'].unique()) + 1)
jokes = range(1, len(train['JID'].unique()) + 1)
all_combinations = [(user, joke) for user in users for joke in jokes]
df_all = pd.DataFrame(all_combinations, columns=['UID', 'JID'])

# находим отсутствующие пары пользователь-шутка
df_missing = df_all.merge(
    train[['UID', 'JID']], 
    on=['UID', 'JID'], 
    how='left', 
    indicator='merged_train'
)
df_missing = df_missing.merge(
    test, 
    on=['UID', 'JID'], 
    how='left', 
    indicator='merged_test'
)
df_missing = df_missing.loc[
    (df_missing['merged_test'] != 'both') & 
    (df_missing['merged_train'] != 'both')
]
df_missing = df_missing.drop(columns=['merged_train', 'merged_test'])

test = pd.concat([test, df_missing], ignore_index=True)

# предикт рейтингов
test['Rating'] = test[['UID', 'JID']].apply(
    lambda x: svd.predict(x[0], x[1], verbose=False).est, 
    axis = 1
)

result, _ = get_top_jokes(test, 10)

# сохранение результата в CSV файл
result_df = pd.DataFrame(result, columns=['UID', 'top_rec'])
result_df.to_csv('output.csv', index=False)