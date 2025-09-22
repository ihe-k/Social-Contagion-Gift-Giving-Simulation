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
git clone https://github.com/your-username/Social-Contagion-Gift-Giving-Simulation.git
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
As chronic illness care shifts from institutions to home and community-based services (HCBS), a major research question emerges: How might decentralised platforms like Airbnb be leveraged or regulated to expand access to care for people living in temporary or unstable housing?

This inquiry builds on this project by embedding care access, mobility constraints and health behaviour dynamics within the real-world platforms, neighborhoods and transportation infrastructure systems that shape life for chronically ill individuals experiencing housing precarity.

Research could focus on exploring the ways housing context, mobility constraints and digital platform ecosystems could impact the accessibility, quality and continuity of HCBS for people with chronic illness residing in temporary accommodations (e.g., shelters, motels or short-term rentals).

### Research Directions
1. Geospatial Mapping of HCBS and Housing Instability:  
* Areas with high turnover or unstable housing that are identified using census and housing market data, such as neighbourhoods with high Airbnb activity, could be mapped alongside housing instability. These transient populations may be more likely to rely on short-term rentals like Airbnb, often due to the instability of longer-term housing options.
* Mapping Airbnb locations alongside HCBS providers could highlight neighbourhoods where Airbnb is more prevalent but healthcare services are sparse. In such regions, guests who rely on short-term stays may find it harder to access formal care, especially if they are located far from established services. This creates an opportunity to identify whether short-term rental density correlates with care deserts and where temporary residents may face challenges in accessing care.

2. Proximity Mapping and Network Connectivity
* In areas with high turnover where Airbnb stays are concentrated but the formal healthcare network is weak, informal care networks (e.g., faith-based health outreach, mutual aid groups, social media networks or street medicine teams) may grow in importance.
* Analysing the proximity between Airbnb listings and care providers can reveal gaps in care delivery as guests in Airbnb rentals may be more likely to rely on local informal care systems or community-driven services, especially in areas with limited HCBS access.
* Mapping these informal systems and Airbnb’s influence on local networks can show mismatches in how care is delivered and whether Airbnb clusters facilitate access to these support systems.

3. Mobility Constraints and Access to Care  
* Affordability vs Mobility: In areas with Airbnb prevalence, affordability often becomes a key consideration for guests. Airbnb guests, especially those in low-cost areas, may face mobility challenges due to poor infrastructure, limiting access to public transportation or essential services. The affordability of Airbnb rentals in areas with inadequate healthcare infrastructure could exacerbate mobility constraints for temporary residents. These guests might be more reliant on local social media groups or neighbourhood-based care, such as mobile clinics, as a means of coping with limited access to healthcare.
* Overcrowding: In cities where Airbnb listings are dense, overcrowding might affect public infrastructure and impair residents from navigating spaces. Poor or unsafe infrastructure, combined with overcrowded streets due to tourists, could create physical mobility barriers. This impacts access to healthcare and could influence whether guests book accommodations in areas with better accessibility or in places where healthcare is harder to reach (thus promoting a reliance on informal care).

4. Sociocultural and Health System Variables
Future iterations can also incorporate sociocultural variables that further influence online health information sharing and access to care.  This will help to broaden the scope of the research, particularly in understanding how digital ecosystems, platforms like Airbnb as well as social networks intersect with health systems. These might include:
* Ethnicity and Nationality: Cultural context deeply affects health beliefs and behaviours. Including demographic data (where ethically sourced and privacy-compliant) could reveal region-specific or ethnically patterned contagion dynamics.
* Religious and Cultural Proscriptions: Religious norms and cultural taboos often guide what kind of health information individuals consider acceptable to share especially regarding topics like mental health, sexual health or vaccination.
* Language and Regional Vernacular: Language usage patterns may reflect underlying cultural values and affect how sentiment as well as ideology is expressed and interpreted.
* Cross-Cultural Contagion: Modelling how information crosses cultural boundaries (e.g., Western vs. Eastern health ideologies) could help predict global misinformation or health behaviour patterns.
* Community-Specific Homophily: Other in-group dynamics, in addition to gender, such as ethnicity-based homophily may also affect contagion probability.

Incorporating these variables will require ethically sourced datasets, potentially leveraging unsupervised learning or transfer learning on large multilingual datasets as well as adapting the sentiment/ideology classifier to better handle cultural nuance and context-specific meaning.

### Platforms and the Future of Distributed Care
Building on precedents like Papa, DispatchHealth and Uber Health, this work could explore speculative models where platform-mediated services help fill care gaps in neighbourhoods with high housing turnover, particularly in the context of vulnerable as well as transient populations.  These speculative models can integrate not only geospatial mobility but also sociocultural factors that influence healthcare access and information diffusion. For example:
* Gig-dispatched care coordinated through app-based platforms (e.g., visiting nurses, paramedics, mental health support workers, dental services, medication delivery) may significantly enhance the community care model.  This includes on-demand emergency care, such as the management of chronic conditions, minor emergencies, on-the-spot diagnostics, health screenings, vaccinations or health check-ups.
* Airbnb hosts or listings as potential centres in distributed care networks (particularly in care deserts or recovery-focused housing).
* Cultural and Ethnic Considerations: Distributed care services could also consider ethnic or cultural norms that affect care delivery. For instance, religious or cultural proscriptions might influence what type of health information or services are shared in these communities.

This research could explore how platforms like Airbnb may be leveraged for temporary housing as well as distributed healthcare delivery to address the needs of those with chronic illness who may be underserved by traditional healthcare models.

### AI Integration
* Geospatial ML models that identify emerging care deserts and predict HCBS demand in areas of high housing turnover.
* Recommender systems to match individuals in short-term housing with nearby services, peer support or mobile clinics while considering sociocultural and health belief variables.  For example, some populations may prefer healthcare services that align with their cultural variables (e.g., alternative care options).
* LLM-driven agents in simulation models could simulate how individuals facing houring instability make health decisions.  This would include replicating how sociocultural factors, such as religious beliefs or ethnic identities) shape their search for information, interpret health messaging or decide between care options.
* Integrating wearable smart devices that are able to track data continuously over time could offer predictive insights into a user's health when paired with AI-powered apps and alert an on-demand emergency care service when a user's metrics are outside of the normal range.  For example, an app might predict a potential flare-up of a chonic condition and prompt an individual to take their medication, schedule a mobile consultation, trigger an ambulance dispatch team or make adjustments to their lifestyle.  This could also complement urgent treatment centres by improving access to care within community settings, particularly for those unable to access fixed-location services due to mobility limitations, by bridging the gap between preventative care and the need for emergency intervention.

### Example Usage
* Social Network Simulation: After setting up the environment, the network is simulated with users sharing information based on their health conditions, ideologies and social ties. This will lead to a dynamic contagion process where information spreads through the network.
* Real-Time Visualisation: The Streamlit dashboard provides a visual representation of the network, highlighting which users are sharing information and how it spreads across different ideological groups and health conditions.
* Health Information Diffusion: Using real-time data from RSS feeds, the model predicts how information will spread among users with varying health conditions and ideologies.

These AI-driven models may also be able to predict how informal care networks evolve in areas with high housing turnover and whether short-term rental platforms like Airbnb may be utilised to strengthen these systems.

## Industry and Public Health Applications
* Social Media and Platform Analytics: Social media platforms or content platforms (e.g., X, YouTube, podcast networks) may use this model to better understand how health-related content spreads through digital communities, including how sociocultural dynamics like ethnicity and religion affect content sharing. Insights on sentiment-driven behaviour and ideological alignment enable more nuanced content recommendation systems and misinformation detection strategies, particularly in communities with high levels of housing instability.
* Behavioural and Market Insights: By modelling how users with different health conditions, ideologies and sociocultural attributes interact, this simulation may inform targeted marketing, health campaign design and consumer segmentation in industries such as digital wellness, telehealth and preventive care. This app highlights how homophily and cross-ideological exposure impact virality as well as trust in health information which are relevant behavioural insights to both commercial and public messaging.
* Public Health Systems: This tool offers an innovative framework to simulate the spread of accurate or misleading health information across different social configurations in public health systems, particularly in the context of chronic disease management, vaccination uptake, or mental health outreach where communities have limited trust in traditional healthcare channels. Incorporating how sociocultural factors and housing instability exposes vulnerable populations (e.g., those living in unstable housing or cultural enclaves) specific health information can help inform public health policy and health communication strategies for diverse groups.
* Real-Time Decision-Making: Organisations like Microsoft or Google can integrate this model into digital health analytics, recommendation engines or real-time monitoring systems to anticipate user responses to health content, adapt interfaces dynamically and personalise intervention strategies. The ability of the app to reflect demographic and ideological nuances allows for more responsive, culturally-sensitive systems
* Short-Term Rental Platforms & Distributed Care Innovation: Platforms like Airbnb may indirectly benefit from this model by understanding how housing instability, geospatial mobility and informal support care networks impact access to care, particularly in care deserts or areas with high Airbnb density. Insights from the model may help short-term rental platforms identify areas with vulnerable populations (e.g., individuals with chronic illness or experiencing housing precarity) and facilitate the design of distributed care models like mobile clinics or gig-based care delivery. Additionally, integrating Airbnb hosts as part of a broader care ecosystem could provide new opportunities for temporary housing to be utilised for care coordination, leveraging informal support systems as well as community-based health initiatives (especially for populations experiencing housing precarity or displacement).

## Contact and Contributing
Contact: For collaboration or inquiries, please contact me here. Contributions are welcome. Feel free to open issues or create pull requests.

## Acknowledgments
* Libraries: Special thanks to the creators of scikit-learn, TextBlob, NetworkX and Streamlit for providing powerful tools that made this project possible.
* OpenAI: For inspiration in understanding AI-driven user behaviour prediction models.
