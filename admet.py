import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

predictor = ADMETModel()

# Custom objects dictionary
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load pre-trained neural network model with custom objects
model = tf.keras.models.load_model("neural_network_model.h5", custom_objects=custom_objects)

# Load dataset with existing points
data = pd.read_csv("Sdmet_predictions.csv")  # Assumes dataset has F0 and F1
scaler = StandardScaler()
features = [col for col in data.columns if col not in ['SMILES', 'F0', 'F1']]
F_Features = ['F0', 'F1']
data[features] = scaler.fit_transform(data[features])

# Function to generate molecular images
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(100, 100))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dash/')

app.layout = html.Div([
    html.H1("SMILES ADMET & Neural Network Analysis"),
    dcc.Input(id='smiles-input', type='text', placeholder="Enter SMILES String"),
    dcc.Input(id='cluster-input', type='number', placeholder="Enter number of clusters (K)", min=1, step=1, value=3),
    dcc.Dropdown(
        id='clustering-type',
        options=[
            {'label': 'F_Features', 'value': 'F_Features'},
            {'label': 'ADMET Features', 'value': 'ADMET_Features'}
        ],
        value='ADMET_Features',
        placeholder="Select Clustering Type"
    ),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='physicochemical-properties'),
    html.Div(id='absorption-distribution'),
    html.Div(id='metabolism'),
    html.Div(id='elimination'),
    html.Div(id='toxicity'),
    html.Div(id='related-chemicals'),
    html.Div(id='related-chemicals_admet'),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    [dash.Output('physicochemical-properties', 'children'),
     dash.Output('absorption-distribution', 'children'),
     dash.Output('metabolism', 'children'),
     dash.Output('elimination', 'children'),
     dash.Output('toxicity', 'children'),
     dash.Output('related-chemicals', 'children'),
     dash.Output('related-chemicals_admet', 'children'),
     dash.Output('scatter-plot', 'figure')],
    [dash.Input('predict-button', 'n_clicks')],
    [dash.State('smiles-input', 'value'), dash.State('cluster-input', 'value'), dash.State('clustering-type', 'value')]
)
def process_smiles(n_clicks, smiles, k, clustering_type):
    if not smiles:
        return "Enter a SMILES string", "", "", "", "", "", px.scatter()
    
    # Predict ADMET properties
    admet_dict = predictor.predict(smiles)  # Assume returns a dictionary of key-value pairs
    admet_df = pd.DataFrame([admet_dict])
    
    # Normalize input and predict F0, F1
    admet_scaled = scaler.transform(admet_df[features])
    f0_f1 = model.predict(admet_scaled)
    f0, f1 = f0_f1[0]
    
    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(data[['F0', 'F1']])
    distances, indices = neighbors.kneighbors(np.array([[f0, f1]]))
    neighbor_points = data.iloc[indices[0]]

    # Find nearest neighbors
    neighbors_admet = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(data[features])
    distances_admet, indices_admet = neighbors_admet.kneighbors(np.array(admet_scaled))
    neighbor_points_admet = data.iloc[indices_admet[0]]


    # Select clustering features based on dropdown choice
    clustering_features = ['F0', 'F1'] if clustering_type == 'F_Features' else features
    
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[clustering_features])

    related_chemicals_html = html.Div([
        html.H3("Structurally Related Chemicals"),
        html.Div([
            html.Div([
                html.Img(src=smiles_to_image(s), style={'width': '100px', 'height': '100px'}),
                html.P(s)
            ], style={'display': 'inline-block', 'margin': '10px'}) for s in neighbor_points.SMILES
        ])
    ])

    related_chemicals_html_admet = html.Div([
        html.H3("ADMET-Wise Related Chemicals"),
        html.Div([
            html.Div([
                html.Img(src=smiles_to_image(s), style={'width': '100px', 'height': '100px'}),
                html.P(s)
            ], style={'display': 'inline-block', 'margin': '10px'}) for s in neighbor_points_admet.SMILES
        ])
    ])
    
    # Organizing ADMET outputs into six sections
    sections = {
        "physicochemical-properties": ['molecular_weight', 'logP', 'hydrogen_bond_acceptors', 'hydrogen_bond_donors', 'Lipinski', 'QED', 'stereo_centers', 'tpsa'],
        "absorption-distribution": ['HIA_Hou', 'BBB_Martins', 'Bioavailability_Ma', 'PAMPA_NCATS', 'Pgp_Broccatelli'],
        "metabolism": ['CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith', 'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Substrate_CarbonMangels', 'CYP3A4_Veith'],
        "elimination": ['Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'Half_Life_Obach'],
        "toxicity": ['hERG', 'Carcinogens_Lagunin', 'ClinTox', 'DILI', 'LD50_Zhu', 'Skin_Reaction']
    }
    
    html_outputs = {}
    for section, keys in sections.items():
        html_outputs[section] = html.Div([
            html.H3(section.replace('-', ' ').title()),
            html.Table([
                html.Tr([html.Th(key), html.Td(round(admet_dict.get(key, 0), 3))]) for key in keys if key in admet_dict
            ])
        ])
    
    # Create scatter plot with nearest neighbors and emphasize predicted F0, F1
    scatter_fig = px.scatter(data, x='F0', y='F1', color='Cluster', title='Predicted Point, Clusters & Nearest Neighbors')
    scatter_fig.add_trace(px.scatter(neighbor_points, x='F0', y='F1', color_discrete_sequence=['blue']).data[0])
    scatter_fig.add_trace(px.scatter(x=[f0], y=[f1], color_discrete_sequence=['red'], size=[10]).data[0])
    
    return html_outputs['physicochemical-properties'], html_outputs['absorption-distribution'], html_outputs['metabolism'], html_outputs['elimination'], html_outputs['toxicity'], related_chemicals_html, related_chemicals_html_admet, scatter_fig

if __name__ == '__main__':
    app.run_server()