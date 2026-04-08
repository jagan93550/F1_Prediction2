# src/app.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.predict import predict_by_circuit

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
        """Simple frontend to input a circuit name and see podium predictions."""

        return """<!doctype html>
<html>
    <head>
        <meta charset='utf-8' />
        <title>F1 Podium Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; }
            h1 { text-align: center; }
            form { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; }
            input[type=text] { width: 100%; padding: 8px; margin-bottom: 12px; }
            button { padding: 8px 16px; cursor: pointer; }
            table { width: 100%; border-collapse: collapse; margin-top: 16px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f4f4f4; }
            #error { color: red; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>F1 Podium Prediction</h1>
        <form id="predict-form">
            <label for="circuit">Circuit name</label>
            <input id="circuit" name="circuit" type="text" placeholder="e.g. monza, silverstone" required />
            <button type="submit">Predict podium</button>
        </form>
        <div id="error"></div>
        <table id="results" style="display:none;">
            <thead>
                <tr><th>Position</th><th>Driver</th><th>Score</th></tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            const form = document.getElementById('predict-form');
            const errorDiv = document.getElementById('error');
            const table = document.getElementById('results');
            const tbody = table.querySelector('tbody');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                errorDiv.textContent = '';
                table.style.display = 'none';
                tbody.innerHTML = '';

                const circuit = document.getElementById('circuit').value.trim();
                if (!circuit) {
                    errorDiv.textContent = 'Please enter a circuit name.';
                    return;
                }

                try {
                    const resp = await fetch(`/predict/circuit/${encodeURIComponent(circuit)}`);
                    if (!resp.ok) {
                        const text = await resp.text();
                        throw new Error(text || `Request failed with status ${resp.status}`);
                    }
                    const data = await resp.json();

                    if (!Array.isArray(data) || data.length === 0) {
                        errorDiv.textContent = 'No predictions available for this circuit.';
                        return;
                    }

                    data.forEach((row, idx) => {
                        const tr = document.createElement('tr');
                        const pos = document.createElement('td');
                        pos.textContent = idx + 1;
                        const name = document.createElement('td');
                        name.textContent = row.name || 'Unknown';
                        const score = document.createElement('td');
                        score.textContent = row.pred_score != null ? row.pred_score.toFixed(4) : '';
                        tr.appendChild(pos);
                        tr.appendChild(name);
                        tr.appendChild(score);
                        tbody.appendChild(tr);
                    });

                    table.style.display = '';
                } catch (err) {
                    console.error(err);
                    errorDiv.textContent = 'Error: ' + err.message;
                }
            });
        </script>
    </body>
</html>"""


@app.get("/predict/circuit/{circuit_name}")
def predict(circuit_name: str):
        """API endpoint returning top 3 predicted drivers for a circuit."""
        result = predict_by_circuit(circuit_name)
        return result.to_dict(orient="records")