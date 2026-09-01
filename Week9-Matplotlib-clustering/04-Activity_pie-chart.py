import matplotlib.pyplot as plt

# Given percentages
apple = 30
banana = 25
grape = 20
total_students = 100

# Calculate the percentage for oranges
orange = 100 - (apple + banana + grape)

# Data for the pie chart
labels = ['Apple', 'Banana', 'Grape', 'Orange']
sizes = [apple, banana, grape, orange]
colors = ['red', 'yellow', 'purple', 'orange']

# Create pie chart
fig = plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Favorite Fruits of 100 Students")
plt.show()
fig.savefig("./Pie.jpg")

# Print the calculated percentage for oranges
print(f"Percentage of students who prefer oranges: {orange}%")
