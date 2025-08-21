import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import random
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import feedparser
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
