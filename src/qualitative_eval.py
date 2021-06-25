import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

top = 20
root_dir = "../data/CUB_200_2011"
val_wrong_count = [
  1,1,0,4,1,2,0,3,3,1,4,0,1,1,2,1,0,0,1,0,3,2,4,3,9,2,6,0,6,9,1,3,8,1,1,0,
  5,3,9,6,4,2,3,4,3,1,1,0,9,3,2,1,0,4,0,1,1,2,4,6,1,10,0,4,5,10,2,1,2,0,4,4,
  1,2,0,0,1,1,4,0,4,0,1,0,1,1,1,3,1,1,7,0,2,0,0,3,7,4,5,0,0,6,3,1,5,0,3,1,
  3,1,8,4,4,0,3,2,4,3,3,2,6,2,4,3,7,5,7,4,5,7,1,2,2,1,5,2,3,4,3,0,1,3,4,6,
  2,4,3,0,2,4,3,2,5,4,4,2,2,3,4,1,5,2,3,0,1,0,2,1,1,5,3,1,4,4,5,1,0,2,4,1,
  2,1,4,4,0,1,0,0,2,1,1,1,2,1,1,3,6,2,5,0
]
print(len(val_wrong_count))
val_wrong_count = np.array(val_wrong_count)
top_wrong_class_idx = np.argsort(val_wrong_count)[-top:]
top_wrong_count = val_wrong_count[top_wrong_class_idx]
print(top_wrong_class_idx)
print(top_wrong_count)

classes = pd.read_csv(
            os.path.join(root_dir, 'classes.txt'), 
            sep=' ', 
            names=['class_id', 'class_name'])
top_wrong_classes = classes.iloc[top_wrong_class_idx]
top_wrong_class_names = list(top_wrong_classes['class_name'])
top_wrong_class_names_without_id = []
for name in top_wrong_class_names:
  new_name = name[4:]
  top_wrong_class_names_without_id.append(new_name)
print(top_wrong_class_names_without_id)

# Horizontal Bar Chart
# fig = plt.figure(figsize=(5, 2))
# names = top_wrong_class_names_without_id
# error_count = top_wrong_count
# plt.barh(names,error_count)
# plt.title('Top 25 Classes with Wrong Classification')
# plt.ylabel('Class Name')
# plt.xlabel('Number of Wrong Images Classified')
# plt.show()

# Pie Chart
def absolute_value(val):
    a  = int(np.round((val/100)*12, 0))
    return a

# labels = 'Correct', 'California_Gull', 'Laysan_Albatross', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Glaucous_winged_Gull'
# sizes = [2,4,2,2,1,1]
# explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Western_Gull Validation Incorrect Classification')
# plt.show()

labels = 'Correct', 'California_Gull', 'Ring_billed_Gull', 'Ring_billed_Gull', 'Western_Gull', 'Slaty_backed_Gull'
sizes = [2,4,3,1,1,1]
explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Herring_Gull Validation Incorrect Classification')
plt.show()