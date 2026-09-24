import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load grocery transaction dataset
data = pd.read_csv("groceries.csv", header=None)

# Convert data into transactions
transactions = []

for i in range(len(data)):
    items = data.iloc[i].dropna().tolist()
    transactions.append(items)

# Convert transactions into one-hot encoded format
te = TransactionEncoder()
encoded_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_data, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.01,
    use_colnames=True
)

print("FREQUENT ITEMSETS")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.2
)

print("\nASSOCIATION RULES")
print(
    rules[
        ["antecedents", "consequents",
         "support", "confidence", "lift"]
    ].sort_values("lift", ascending=False).head(10)
)
