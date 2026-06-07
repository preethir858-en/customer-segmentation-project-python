import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load dataset

data = pd.read_csv("customer_data.csv")

# Convert Gender to numeric

encoder = LabelEncoder()

data["Gender"] = encoder.fit_transform(
    data["Gender"]
)

# Select demographic + behavior features

X = data[
[
"Age",
"Gender",
"AnnualIncome",
"PurchaseAmount",
"PurchaseFrequency",
"SpendingScore"
]
]

# Scale data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Clustering

kmeans = KMeans(
n_clusters=3,
random_state=42,
n_init=10
)

data["Segment"] = kmeans.fit_predict(
X_scaled
)

# Save output

data.to_csv(
"segmented_customers.csv",
index=False
)

print(data)

# Visualization

plt.figure(figsize=(8,6))

scatter = plt.scatter(
data["AnnualIncome"],
data["PurchaseAmount"],
c=data["Segment"]
)

plt.xlabel(
"Annual Income"
)

plt.ylabel(
"Purchase Amount"
)

plt.title(
"Customer Segmentation"
)

plt.show()