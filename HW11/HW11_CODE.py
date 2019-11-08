#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np
from pulp import *


# In[163]:


diet = pd.read_csv("data 15.2/diet.csv")
diet_large = pd.read_csv("data 15.2/diet_large.csv")


# # Pre-Processing

# In[95]:


diet


# In[96]:


diet.rename(columns = {'Price/ Serving':'Price_Serving'}, inplace=True)
    


# In[97]:


diet.head()


# In[98]:


diet['Price_Serving'] = diet['Price_Serving'][:64].apply(lambda x: str(x)[1:])


# In[99]:


diet


# In[100]:


diet['Price_Serving'] = diet['Price_Serving'][:64].apply(lambda x: float(x))


# In[101]:


diet


# In[102]:


def food_dict_maker(colSeries, colname):
    food_colname = list(zip(diet['Foods'][:64], colSeries[:64]))
    food_colname_dict = {}
    for index, food_colname in enumerate(food_colname):
        food_colname_dict[food_colname[0]] = food_colname[1]
    return food_colname_dict
    


# In[103]:


food_price = food_dict_maker(diet['Price_Serving'], 'price')
food_calories = food_dict_maker(diet['Calories'], 'calories')
food_cholesterol = food_dict_maker(diet['Cholesterol mg'], 'cholesterol')
food_fat = food_dict_maker(diet['Total_Fat g'], 'fat')
food_sodium = food_dict_maker(diet['Sodium mg'], 'sodium')
food_carbohydrates = food_dict_maker(diet['Carbohydrates g'], 'carbohydrates')
food_dietary_fiber = food_dict_maker(diet['Dietary_Fiber g'], 'dietary_fiber')
food_protein = food_dict_maker(diet['Protein g'], 'protein')
food_vit_A = food_dict_maker(diet['Vit_A IU'], 'vit_A')
food_vit_C = food_dict_maker(diet['Vit_C IU'], 'vit_C')
food_calcium = food_dict_maker(diet['Calcium mg'], 'calcium')
food_iron = food_dict_maker(diet['Iron mg'], 'iron')


# In[104]:


cols = list(diet)[3:]
min_nutrients = [diet[col][65] for col in cols]
max_nutrients = [diet[col][66] for col in cols]


# In[105]:


ingredients = [ingr for ingr in diet['Foods'][:64]]


# In[106]:


prob = LpProblem("Diet Problem Simple", LpMinimize)


# In[107]:


ingredient_vars = LpVariable.dicts("Ingr", ingredients,0)


# In[108]:


prob += lpSum([food_price[i]*ingredient_vars[i] for i in ingredients]), "Total Cost of Ingredients"


# In[109]:


food_dicts = [food_calories, food_cholesterol, food_fat, food_sodium, food_carbohydrates, food_dietary_fiber, food_protein, food_vit_A, food_vit_C, food_calcium, food_iron]


# In[110]:


requirementList = [('CaloriesMinRequirement', 'CalorieMaxRequirement'),
                    ('CholesterolMinRequirement', 'CholesterolMaxRequirement'),
                  ('FatMinRequirement', 'FatMaxRequirement'),
                  ('SodiumMinRequirement','SodiumMaxRequirement'),
                  ('CarbohydratesMinRequirement', 'CarbohydratesMaxRequirement'),
                  ('DietaryFiberMinRequirement','DietaryFiberMaxRequirement'),
                  ('ProteinMinRequirement', 'ProteinMaxRequirement'),
                  ('VitAMinRequirement', 'VitAMaxRequirement'),
                  ('VitCMinRequirement', 'VitCMaxRequirement'),
                  ('CalciumMinRequirement', 'CalciumMaxRequirement'),
                  ('IronMinRequirement', 'IronMaxRequirement')]


# In[111]:


for i, food_dict in enumerate(food_dicts):
    prob += lpSum([food_dict[j]*ingredient_vars[j] for j in ingredients]) >= min_nutrients[i], requirementList[i][0]
    prob += lpSum([food_dict[j]*ingredient_vars[j] for j in ingredients]) <= max_nutrients[i], requirementList[i][1]


# In[112]:


prob.writeLP("DietSimple.lp")


# In[113]:


prob.solve()


# In[114]:


print("Status:", LpStatus[prob.status])


# In[115]:


for v in prob.variables():
    if(v.varValue != 0.0):
        print(v.name, " = ", v.varValue)


# In[116]:


value(prob.objective)


# ## 1 (B)
# 

# In[117]:


bin_ingr_vars = pulp.LpVariable.dicts('binary_', ingredients, lowBound = 0, upBound = 1, cat = LpBinary)


# In[118]:


diet_model = LpProblem("Diet Model Complicated", LpMinimize)


# In[119]:


diet_model += lpSum([food_price[i]*ingredient_vars[i] for i in ingredients]), "Total Cost of Ingredients"


# ### Constraint A

# In[120]:


for i in ingredients:
    diet_model += ingredient_vars[i] >= 0.1*bin_ingr_vars[i]
       


# ### Constraint B

# In[121]:


diet_model += bin_ingr_vars['Celery, Raw'] + bin_ingr_vars['Frozen Broccoli'] <= 1


# ### Constraint C

# In[122]:


meats = ['Roasted Chicken', 'White Tuna in Water', 'Taco', 'Malt-O-Meal,Choc', 
 'Pork', 'Hamburger W/Toppings', 'Vegetbeef Soup', 'Splt Pea&Hamsoup',
 'Neweng Clamchwd','Hotdog, Plain','Pizza W/Pepperoni','New E Clamchwd,W/Mlk']


# In[123]:


diet_model += lpSum([bin_ingr_vars[meat] for meat in meats]) >= 3, " Meat Constraint"


# In[124]:


for i, food_dict in enumerate(food_dicts):
    diet_model += lpSum([food_dict[j]*ingredient_vars[j] for j in ingredients]) >= min_nutrients[i], requirementList[i][0]
    diet_model += lpSum([food_dict[j]*ingredient_vars[j] for j in ingredients]) <= max_nutrients[i], requirementList[i][1]


# In[125]:


for i in range(0,len(max_nutrients)):
    for j in ingredients:
        diet_model += ingredient_vars[j] <= max_nutrients[i]*bin_ingr_vars[j]


# In[126]:


diet_model.writeLP("DietMoreConstraints.lp")


# In[127]:


diet_model.solve()


# In[128]:


print("Status:", LpStatus[diet_model.status])


# In[129]:


print(value(diet_model.objective))


# In[130]:


for v in diet_model.variables(): 
    if(v.varValue != 0.0):
        print(v.name, " = ", v.varValue)


# ## Diet Large Model

# In[166]:


diet_large.fillna(0, inplace=True)


# In[167]:


def diet_large_dict_maker(colSeries, colname):
    food_colname = list(zip(diet_large['Long_Desc'], colSeries[:7146]))
    food_colname_dict = {}
    for index, food_colname in enumerate(food_colname):
        food_colname_dict[food_colname[0]] = food_colname[1]
    return food_colname_dict
    


# In[170]:


diet_large_list = []
for col in list(diet_large)[1:]:
    diet_large_list.append(diet_large_dict_maker(diet_large[col], col))


# In[172]:


diet_large_min_nutrients = [diet_large[col][7146] for col in list(diet_large)[1:]]
diet_large_max_nutrients = [diet_large[col][7147] for col in list(diet_large)[1:]]


# In[175]:


diet_large_ingredients = [ingr for ingr in diet_large['Long_Desc'][:7146]]


# In[176]:


diet_large_model = LpProblem("Diet Large Cholesterol Problem", LpMinimize)


# In[177]:


diet_large_ingredient_vars = LpVariable.dicts("Ingr", diet_large_ingredients,0)


# In[178]:


diet_large_cholesterol = diet_large_dict_maker(diet_large['Cholesterol'], 'Cholesterol')


# In[193]:


[diet_large_cholesterol[i] for i in diet_large_ingredients]


# In[179]:


diet_large_model += lpSum([diet_large_cholesterol[i]*diet_large_ingredient_vars[i] for i in diet_large_ingredients]), "Total Cholesterol of Ingredients"


# In[181]:


for i, food_dict in enumerate(diet_large_list):
    diet_large_model += lpSum([food_dict[j]*diet_large_ingredient_vars[j] for j in diet_large_ingredients]) >= diet_large_min_nutrients[i]
    diet_large_model += lpSum([food_dict[j]*diet_large_ingredient_vars[j] for j in diet_large_ingredients]) <= diet_large_max_nutrients[i]


# In[182]:


diet_large_model.writeLP("DietLargeCholesterol.lp")


# In[195]:


diet_large_model.solve()


# In[197]:


print("Status:", LpStatus[diet_large_model.status])


# In[198]:


for v in diet_large_model.variables():
    if(v.varValue != 0.0):
        print(v.name, " = ", v.varValue)


# In[200]:


print(value(diet_large_model.objective))

