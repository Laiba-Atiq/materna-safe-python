import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def createBoxplot(ax, data, title):
   sb.boxplot(y=data, ax=ax)
   ax.set_title(title, fontsize=11)

def createBarChart(ax, data, title, cat1, cat2):
  count = data.value_counts()
  bars = ax.bar([cat1,cat2], count.values, color=["steelblue", "tomato"], width=0.4)

  ax.set_title(title, pad=10, fontsize=11)
  ax.set_xlabel("Category", fontsize=9)
  ax.set_ylabel("Count", fontsize=9)

def createScatterPlot(ax, col1,col2, cat1, cat2):
  ax.scatter(col1, col2)
  ax.set_title(f"{cat1} vs {cat2}")
  ax.set_xlabel(cat1)
  ax.set_ylabel(cat2)

def visualisation(df):
  numericalCol = ["Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI", "Heart Rate"]
  categoricalCol = ["Previous Complications", "Preexisting Diabetes", "Gestational Diabetes", "Mental Health"]
  targetCol = "Risk Level"

  categoricalLabels = {
       "Previous Complications": ("No Complications", "Has Complications"),
       "Preexisting Diabetes":   ("No Diabetes", "Has Diabetes"),
       "Gestational Diabetes":   ("No Gest. Diabetes", "Has Gest. Diabetes"),
       "Mental Health":          ("No Issues", "Has Issues"),
       }
    
  #numerical features  
  fig, axes = plt.subplots(2, 4, figsize=(14, 8))
  axes = axes.flatten()
  
  fig.suptitle("Numerical Features", fontsize=16)
  
  for i, col in enumerate(numericalCol):
    createBoxplot(axes[i], df[col], col)
    
  axes[-1].axis("off")
  
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.show()

  #categorical features
  fig, axes = plt.subplots(2, 2, figsize=(14, 9))
  axes = axes.flatten()

  fig.suptitle("Categorical Features", fontsize=13)
  for i, col in enumerate(categoricalCol):
    createBarChart(axes[i], df[col],col,*categoricalLabels[col])

  plt.tight_layout()
  plt.show()

  #target column
  target_counts = df[targetCol].value_counts()
  plt.bar(target_counts.index, target_counts.values, color=["steelblue", "tomato"])
  plt.title("Target Column: Risk Level")
  plt.xlabel("Category")
  plt.ylabel("Count")

  plt.tight_layout()
  plt.show()

  #relations between some numerical values
  pairs = [
    ("Systolic BP", "Diastolic"),  #strong positive
    ("Age", "Systolic BP"),        #moderate positive
    ("BMI", "BS"),                 #moderate positive
    ("Body Temp", "Heart Rate"),   #weak/no correlation
    ("Age", "BMI"),                #weak
    ("Age", "Heart Rate")          #weak negative
    ]

  fig, axes = plt.subplots(3, 2, figsize=(14, 9))
  axes = axes.flatten()

  fig.suptitle("Relations between Some Numerical Values", fontsize=13)
  
  for i, (x, y) in enumerate(pairs):
    createScatterPlot(axes[i],df[x], df[y],*pairs[i])
    
  plt.tight_layout()
  plt.show()

  #pair plot
  g = sb.pairplot(df, hue="Risk Level")
  g.fig.suptitle("Pair Plot of Health Dataset", y=1.02)

  plt.show()
