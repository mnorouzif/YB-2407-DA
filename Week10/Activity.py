import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Sample_Data_for_Activity.csv")
print("Print top 5 records:", df.head())

sns.displot(df['Normal_Distribution'], kde=True)


output_path = "./distribution_plot.jpg"
plt.savefig(output_path, format="jpg", dpi=300)