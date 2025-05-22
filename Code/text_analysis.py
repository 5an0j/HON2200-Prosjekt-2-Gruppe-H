import matplotlib.pyplot as plt
import pandas as pd
import string

files = {
    'Kina (3.10)': 'txt/China.txt',
    'Polen (6.85)': 'txt/Poland.txt',
    'USA (7.85)': 'txt/USA.txt',
    'EU/EØS (8.00)': 'txt/EU.txt'
#    'Norway': 'txt/Norway.txt',
#    'Brazil': 'txt/Brazil_summary.txt',
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
palette1 = [ "#299C81", "#6F7BB7", "#9D5C67" ]
palette2 = [ "#B69339", "#B25C3D", "#A6695C", "#4D7856" ]

df = pd.DataFrame(results).T.round(2)
df = df / (df.max()) # normalize


# ax = df.drop(columns=['Utvikling', 'Økonomi/konkurranseevne']).plot(kind='bar', width=0.8, figsize=(10, 8), color=palette)
# plt.title('Nøkkelordfordelinger i nasjonale og regionale KI-strategier')
# plt.ylabel('Frekvens per 1000 ord')
# plt.xticks(rotation=0)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()


ax = df[['Demokrati/rettigheter', 'Etikk/ansvar', 'Overvåkning/kontroll']].plot(kind='bar', width=0.8, figsize=(10, 8), color=palette1)
plt.title('Demokratisk vokabular i KI-strategier')
plt.ylabel('Frekvens per 1000 ord (normalisert)')
plt.xticks(rotation=0)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()


df[['Utvikling', 'Økonomi/konkurranseevne', 'Implementering/utrulling', 'Bærekraft']].plot(kind='bar', width=0.6, figsize=(10, 8), color=palette2)
plt.title('Vokabular tilknyttet risikovillighet i KI-strategier')
plt.ylabel('Frekvens per 1000 ord (normalisert)')
plt.xticks(rotation=0)
plt.legend(fontsize=12, loc='lower left')
plt.tight_layout()
plt.show()


df.to_csv('keyword_results.csv')