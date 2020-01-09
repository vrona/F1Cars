#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# script to generate a json category to label table
import json

cat_to_name = {}

cat_to_name['1'] = {"driver": "Kimi Raikkonen", "team": "Ferrari"}
cat_to_name['2'] = {"driver": "Sebastian Vettel", "team": "Ferrari"}
cat_to_name['3'] = {"driver": "Esteban Ocon", "team": "Force India"}
cat_to_name['4'] = {"driver": "Sergio Perez", "team": "Force India"}
cat_to_name['5'] = {"driver": "Romain Grosjean", "team": "Haas"}
cat_to_name['6'] = {"driver": "Kevin Magnussen", "team": "Haas"}
cat_to_name['7'] = {"driver": "Fernando Alonso", "team": "Mclaren"}
cat_to_name['8'] = {"driver": "Stoffel Vandoorne", "team": "Mclaren"}
cat_to_name['9'] = {"driver": "Valterri Bottas", "team": "Mercedes-Benz"}
cat_to_name['10'] = {"driver": "Lewis Hamilton", "team": "Mercedes-Benz"}
cat_to_name['11'] = {"driver": "Daniel Ricciardo", "team": "Redbull"}
cat_to_name['12'] = {"driver": "Max Verstappen", "team": "Redbull"}
cat_to_name['13'] = {"driver": "Nico Hulkenberg", "team": "Renault"}
cat_to_name['14'] = {"driver": "Carlos Sainz jr", "team": "Renault"}
cat_to_name['15'] = {"driver": "Marcus Ericsson", "team": "Sauber"}
cat_to_name['16'] = {"driver": "Charles Leclerc", "team": "Sauber"}
cat_to_name['17'] = {"driver": "Pierre Gasly", "team": "Toro Rosso"}
cat_to_name['18'] = {"driver": "Brendon Hartley", "team": "Toro Rosso"}
cat_to_name['19'] = {"driver": "Sergey Sirotkin", "team": "Williams"}
cat_to_name['20'] = {"driver": "Lance Stroll", "team": "Williams"}

cat = json.dumps(cat_to_name)
with open("/Users/me/vrona/Project_1_F1_classifier/\
cat_to_nameF1.json", "w") as f:
    f.write(cat)
