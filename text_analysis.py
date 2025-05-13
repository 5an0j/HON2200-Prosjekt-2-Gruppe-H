import matplotlib.pyplot as plt
import pandas as pd
import string

files = {
    'China': 'Strategies/txt/China.txt',
    'EU': 'Strategies/txt/EU.txt',
    'Norway': 'Strategies/txt/Norway.txt',
    'USA': 'Strategies/txt/USA.txt',
    'Brazil': 'Strategies/txt/Brazil_summary.txt',
    'Poland': 'Strategies/txt/Poland.txt'
}

df_keywords = pd.read_excel('keywords.xlsx')

keywords_to_category = {
    column: df_keywords[column].dropna().str.lower().tolist()
    for column in df_keywords.columns
}

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def count_keywords(text, keywords_to_category):
    counts = {}
    text = remove_punctuation(text)
    words_total = len(text.split())

    for category, keywords in keywords_to_category.items():
        count = sum(text.count(keyword.lower()) for keyword in keywords)
        counts[category] = (count / words_total) * 1000
    return counts

results = {}

for region, filename in files.items():
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        results[region] = count_keywords(text, keywords_to_category)

plt.rcParams.update({'font.size': 16})
palette = ['#332288', '#88CCEE', '#44AA99', '#117733', '#DDCC77', '#CC6677', '#AA4499']

df = pd.DataFrame(results).T.round(2)

ax = df.drop(columns=['Utvikling', 'Økonomi/konkurranseevne']).plot(kind='bar', width=0.8, figsize=(10, 8), color=palette)
plt.title('Nøkkelordfordelinger i nasjonale og regionale KI-strategier')
plt.ylabel('Frekvens per 1000 ord')
plt.xticks(rotation=0)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

df[['Utvikling']].plot(kind='bar', width=0.6, figsize=(10, 8), color='#6699CC')
plt.title('Nøkkelordfordelinger i nasjonale og regionale KI-strategier')
plt.ylabel('Frekvens per 1000 ord')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

df[['Økonomi/konkurranseevne']].plot(kind='bar', width=0.6, figsize=(10, 8), color='#749d5c')
plt.title('Nøkkelordfordelinger i nasjonale og regionale KI-strategier')
plt.ylabel('Frekvens per 1000 ord')
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

print(df)