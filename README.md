# Gift Triggered Health Information Contagion Simulation
This project leverages advanced machine learning models and social network analysis to predict user behaviour/information diffusion patterns across platforms. Using sentiment analysis and logistic regression, this work explores the impact of ideologies, health conditions and social contagion on health information sharing.

The goal is to understand how users with varying health conditions, ideologies and behaviours might influence each other in a digital ecosystem. This model can be used to predict the likelihood of health-related information spreading across social networks and help design more effective public health communication strategies as well as targeted interventions.

[Link to App](https://social-contagion-gift-giving-simulation-bys7mmnchm6xnad4tllzxs.streamlit.app/)

## Key Features
* Behavioural Prediction: Predict how users with different ideologies (pro-health, anti-health and neutral) and health conditions (chronic disease and healthy) behave in social contagion scenarios.
* Homophily and Ideology Influence: Users with the same gender (homophily) are more likely to influence each other. Similarly, users sharing the same health-related ideology (pro-health or anti-health) are more likely to spread similar content. The probability of sharing information depends on factors like trustworthiness (sentiment alignment), gender homophily and cross-ideology interaction. Users with chronic diseases or females (in the model) are more likely to share information, as represented by certain bonuses applied to the contagion probability.
* Social Network Simulation: Build a network of users with unique characteristics and simulate information spreading across this network.
* Sentiment and Ideology Classification: Classify user-generated content as either pro-health, anti-health or neutral using sentiment analysis.
* Visualisation: Real-time visualisation of contagion spread over a social network graph using Matplotlib and Streamlit.
* Model Evaluation: Display model accuracy and a classification report in the Streamlit dashboard for better transparency.

## Practical Strategies for Effective Influence and Content Dissemination
Insights from network diagrams and contagion simulations may optimise health information campaigns by:
* Empowering High-Centrality Users:  Identifying users with high betweenness centrality serving as key bridges within the network may be encouraged to share verified and accurate content. Their position enables them to influence multiple communities effectively.
* Engaging with Active Sharers:  Recognising users who have shared content multiple times (large and triggered nodes) as community leaders or ambassadors to promote trustworthy health information may be particularly beneficial for minoritised and hard to reach communities.
* Targeting Key Clusters:  Using the visualisation may identify tightly connected communities. Tailoring messages within these clusters amplifies reach and peer-to-peer influence.
*  Track Contagion Dynamics: Mapping out natural influence routes along network pathways during campaigns in real-time to assess which areas or users are most active (influential) may enable refinement of strategies as well as facilitation of efficient and widespread information diffusion.

## Technologies Used
* Python Libraries: scikit-learn, NetworkX, Matplotlib and TextBlob
* Streamlit: Interactive web interface for visualising contagion spread and model evaluation
* Machine Learning Models: Random Forest Classifier for user behaviour prediction based on sentiment, ideology and health conditions
* Social Network Analysis: Erdős–Rényi model for generating random social networks

## Industry Use Cases
* Social Media Platforms: Social media platforms or podcast platforms may use this model to better understand content virality and sentiment-driven user behaviour, enhancing content recommendation systems.
* Behavioural Insights: By studying the influence of ideology and chronic health conditions on information dissemination, this work can assist in targeted marketing, health campaigns or consumer behaviour analysis.
* Real-Time Decision-Making: Organisations like Microsoft and Google can integrate this model into real-time analytics for digital health platforms, recommendation engines and user behaviour prediction.

## Academic Relevance
* Public Health Systems: This model serves as an innovative tool for understanding information diffusion and social contagion in public health systems, particularly in the context of health misinformation or chronic disease management.
* Behavioural Science Research: This project integrates behavioural science and machine learning, contributing to research on how ideologies and health conditions influence human behaviour in social networks.
* Data Science & AI: The project demonstrates the application of machine learning algorithms (e.g., logistic regression) in predicting social behaviour and potentially may contribute to data science as well as AI research on social media networks.

## Project Structure
```plaintext
gift-contagion-simulation/
├── Code.py  # Main simulation and Streamlit dashboard script
├── requirements.txt                         # List of required Python packages
├── README.md                                # Project overview and instructions
```

## Requirements
Before running the app, make sure you have the following libraries installed:

```plaintext
pip install streamlit networkx matplotlib
```

### Step 1: Clone the Repository
Clone the repository to your local machine:

```plaintext
git clone https://github.com/your-username/health-info-contagion.git
```
### Step 2: Install Required Libraries
Install the necessary Python libraries by running the following command:

```plaintext
pip install -r requirements.txt
```
Make sure your requirements.txt includes:
snscrape
networkx
matplotlib
streamlit

### Step 3: Run the Streamlit App
To launch the Streamlit app, simply run:

```plaintext
streamlit run app.py
```

### Step 4: Open the app in your browser (it will open automatically):
http://localhost:8501

### Code Updates
1. Model Evaluation in Streamlit: To display model accuracy and classification report, include the following code after training the model: 

```plaintext
# Evaluate model performance
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

accuracy = accuracy_score(y_test, y_pred)

# Display model evaluation in Streamlit
st.subheader("Model Evaluation")
st.write(f"Model Accuracy: {accuracy:.2%}")

# Classification report
report = classification_report(y_test, y_pred, output_dict=False)
st.text("Classification Report:")
st.text(report)
```
2. Visualisation Updates: The Streamlit visualisation for the contagion process will remain the same but you can further enhance it by adding labels for clarity on user attributes (e.g., gender or health conditions) and contagion status.

3. Sentiment and Ideology Updates: The analyze_sentiment function can be updated to not only classify sentiment but also map it to ideologies (pro-health, anti-health and neutral). The user profile creation logic in your user_datasection should reflect this approach.

4. Model Interpretation: Optionally, include SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to provide explanations for model predictions which is helpful for both academic transparency and industry adoption.

## Model Improvements and Optimisation
1. Feature Engineering
To enhance the predictive power of the model several features were engineered to capture more complex patterns in the data. These features include:

* Network Centrality Measures:
Rationale: When dealing with network data (e.g., social media or interactions between users), centrality measures like betweenness centrality are important in identifying influential nodes.
Implementation: Betweenness centrality was computed using the NetworkX library and added as a feature to capture the importance of each node in the network.
Example: "Nodes with high betweenness centrality were likely to be more influential in the network."

2. Model Performance Evaluation
After implementing these feature engineering techniques and optimising the hyperparameters, the final model was evaluated based on various metrics like accuracym and F1-score. The model performance improved, particularly especially in handling time-dependent features and network-based data.

## Future Work
While the current model effectively captures ideological and health-related drivers of behaviour, future iterations can incorporate sociocultural variables that further influence online health information sharing. These include:
* Ethnicity and Nationality: Cultural context deeply affects health beliefs and behaviours. Including demographic data (where ethically sourced and privacy-compliant) could reveal region-specific or ethnically patterned contagion dynamics.
* Religious and Cultural Proscriptions: Religious norms and cultural taboos often guide what kind of health information individuals consider acceptable to share especially regarding topics like mental health, sexual health or vaccination.
* Language and Regional Vernacular: Language usage patterns may reflect underlying cultural values and affect how sentiment as well as ideology is expressed and interpreted.
* Cross-Cultural Contagion: Modelling how information crosses cultural boundaries (e.g., Western vs. Eastern health ideologies) could help predict global misinformation or health behaviour patterns.
* Community-Specific Homophily: Other in-group dynamics, in addition to gender, such as ethnicity-based homophily may also affect contagion probability.
Incorporating these variables will require ethically sourced datasets, potentially leveraging unsupervised learning or transfer learning on large multilingual datasets and adapting the sentiment/ideology classifier to better handle cultural nuance and context-specific meaning.

## Example Usage
* Social Network Simulation: After setting up the environment, the network is simulated with users sharing information based on their health conditions, ideologies and social ties. This will lead to a dynamic contagion process where information spreads through the network.
* Real-Time Visualisation: The Streamlit dashboard provides a visual representation of the network, highlighting which users are sharing information and how it spreads across different ideological groups and health conditions.
* Health Information Diffusion: Using real-time data from RSS feeds, the model predicts how information will spread among users with varying health conditions and ideologies.

## Contact and Contributing
Contact: For collaboration or inquiries, please contact me here. Contributions are welcome. Feel free to open issues or create pull requests.

## Acknowledgments
* Libraries: Special thanks to the creators of scikit-learn, TextBlob, NetworkX and Streamlit for providing powerful tools that made this project possible.
* OpenAI: For inspiration in understanding AI-driven user behaviour prediction models.
