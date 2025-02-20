from flask import Flask, request, render_template, jsonify
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs, AllChem, Descriptors
import pandas as pd
import os
import joblib
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI


client = OpenAI(
)


df = pd.read_csv('tsne_results.csv')

# Fit Nearest Neighbors model based on x, y coordinates
coords = df[["FingerPrint_0", "FingerPrint_1"]].values
nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(coords)

app = Flask(__name__)
server = dash.Dash(__name__, server=app, routes_pathname_prefix='/')
fig = px.scatter(df, x="FingerPrint_0", y="FingerPrint_1", hover_data=["SMILES"], title="Chemical Space t-SNE Plot")

server.layout = html.Div([
    html.H1("Chemical Space Visualization"),
    dcc.Input(id='cluster-count', type='number', value=3, min=1, step=1, placeholder="Enter Cluster Count"),
    html.Button('Update Clusters', id='update-button', n_clicks=0),
    dcc.Graph(id='scatter-plot'),
    html.Div(id='click-data', style={'padding': '20px', 'border': '1px solid black', 'margin-top': '20px'})
])

@server.callback(
    dash.Output('scatter-plot', 'figure'),
    [dash.Input('update-button', 'n_clicks')],
    [dash.State('cluster-count', 'value')]
)
def update_clusters(n_clicks, cluster_count):
    if cluster_count is None or cluster_count < 1:
        cluster_count = 12
    
    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['FingerPrint_0', 'FingerPrint_1']])
    
    fig = px.scatter(df, x="FingerPrint_0", y="FingerPrint_1", hover_data=["SMILES"], color=df['cluster'].astype(str), title="Chemical Space t-SNE Plot with Clustering")
    return fig


@server.callback(
    dash.Output('click-data', 'children'),
    [dash.Input('scatter-plot', 'clickData')]
)
def display_click_data(clickData):
    if clickData:
        smiles = clickData['points'][0]['customdata'][0]
        x, y = clickData['points'][0]['x'], clickData['points'][0]['y']
        mol = Chem.MolFromSmiles(smiles)
        
        # Calculate molecular properties
        molecular_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        
        # Find nearest neighbors based on x, y
        coords = df[["FingerPrint_0", "FingerPrint_1"]].values
        nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(coords)
        distances, indices = nbrs.kneighbors([[x, y]])
        nearest_smiles = [df.iloc[i]["SMILES"] for i in indices[0]]
        img_path = "assets/image.png"
        Draw.MolToFile(mol, img_path, size=(300, 300))
        nearest_images = []
        for smile in nearest_smiles:
            mol = Chem.MolFromSmiles(smile)
            paths = "assets/image"+smile+".png"
            Draw.MolToFile(mol, paths, size=(300, 300))
            nearest_images.append(paths)

        return html.Div([
            html.H3(f"Selected Molecule: {smiles}"),
            html.Img(src=img_path),
            html.P(f"Molecular Weight: {molecular_weight}"),
            html.P(f"LogP: {logp}"),
            html.P(f"Number of Hydrogen Donors: {num_h_donors}"),
            html.P(f"Number of Hydrogen Acceptors: {num_h_acceptors}"),
            html.H4("5 Nearest Molecules"),
            html.Ul([html.Li(html.Div([html.Img(src=img, style={'width': '100px', 'height': '100px'}), s])) for s, img in zip(nearest_smiles, nearest_images)])
        ])
    return "Click on a point to see molecular details."


# Feature 1: Molecular Structure & Similarity with Additional Analysis and 3D Visualization
@app.route('/molecule', methods=['GET', 'POST'])
def molecule_analysis():
    if request.method == 'POST':
        smiles = request.form['smiles']
        mol = Chem.MolFromSmiles(smiles)
        img_path = "static/molecule.png"
        Draw.MolToFile(mol, img_path, size=(300, 300))
        
        # Generate 3D coordinates
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d)
        AllChem.UFFOptimizeMolecule(mol_3d)
        mol_block = Chem.MolToMolBlock(mol_3d)
        
        # Calculate molecular properties
        molecular_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        
        return render_template('molecule.html', image_path=img_path, 
                               molecular_weight=molecular_weight, logp=logp, 
                               num_h_donors=num_h_donors, num_h_acceptors=num_h_acceptors,
                               mol_block=mol_block,
                               smiles=smiles)
    return render_template('molecule.html')

# Feature 3: Symptom-based Diagnosis
@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    symptoms = request.form['symptoms']
    model = joblib.load('diagnosis_model.pkl')  # Pre-trained model
    prediction = model.predict([symptoms])
    return jsonify({"diagnosis": prediction[0]})

@app.route('/chat', methods=['POST'])
def chat():
    print(request.json.get('smiles'))
    smiles = request.json.get('smiles')  # Use JSON body instead of query params
    text = smiles
    if not smiles:
        return jsonify({"error": "SMILES input is required"}), 400
    
    try:
        
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            stream=True,
        )
        OUTPUT = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                OUTPUT = OUTPUT + chunk.choices[0].delta.content
        return jsonify({"response": OUTPUT})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Feature 4: File Analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename})
    return jsonify({"error": "No file uploaded"})

if __name__ == '__main__':
    app.run(debug=True, port=6000)