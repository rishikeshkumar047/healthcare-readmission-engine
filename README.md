# Healthcare Readmission Engine

A full-stack machine learning application to predict patient hospital readmission risk and enable proactive care interventions.

## Overview
This system combines: 
- **Backend**: FastAPI REST API serving a trained ML model
- **Frontend**: React web application for care team workflows
- **Data Pipeline**: ETL and model training scripts
- **Docs**: Model card and data dictionary for governance and reproducibility

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 14+
- (Optional) Docker & Docker Compose

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py  # Train the model (creates model.joblib)
python -m uvicorn app:app --reload  # Start API on http://localhost:8000
```

Visit http://localhost:8000/docs for the interactive Swagger API documentation.

### Frontend Setup

```bash
cd frontend
npm install
npm start  # Starts dev server on http://localhost:3000
```

## Project Structure

```
healthcare-readmission-engine/
├── backend/
│   ├── app.py           # FastAPI main application
│   ├── model.joblib     # The trained ML model saved file
│   ├── requirements.txt # Python dependencies
│   └── train_model.py   # Script to ETL data and train the model
├── frontend/            # Standard React Create-React-App structure
│   ├── public/
│   └── src/
│       ├── App.js
│       └── components/
├── data/
│   └── mock_patient_data.csv # Your raw training data
├── docs/
│   ├── data_dictionary.md    # Field descriptions and preprocessing guidance
│   └── model_card.md         # Model governance and reproducibility info
└── README.md
```

## Key Features

- **Risk Prediction**: Scores patients on readmission risk using XGBoost model
- **REST API**: JSON endpoints for batch and real-time predictions
- **Governance**: Comprehensive model card and data dictionary
- **Reproducibility**: Full ETL and training pipeline documented and version-controlled

## Documentation

- **[Model Card](./docs/model_card.md)**: Model purpose, evaluation metrics, limitations, and maintenance guidance
- **[Data Dictionary](./docs/data_dictionary.md)**: Detailed field descriptions, types, and preprocessing steps

## API Endpoints

- `GET /health` — Health check
- `POST /predict` — Single patient prediction
- `POST /predict-batch` — Batch predictions
- `GET /model-info` — Model metadata and version

## Model Performance

See [docs/model_card.md](./docs/model_card.md#4-metrics) for evaluation metrics on test data.

## Development

### Running Tests
```bash
cd backend
pytest tests/
```

### Code Style
- Backend: Follow PEP 8; format with `black` and lint with `pylint`
- Frontend: ESLint configured in `frontend/`

## Deployment

### Docker
```bash
docker-compose up
```

This will start both backend (port 8000) and frontend (port 3000).

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request with a clear description
4. Ensure all tests pass

## Ethical Considerations

This model makes predictions but does not replace human clinical judgment. Always review model scores in context with clinical expertise. See [docs/model_card.md](./docs/model_card.md#6-ethical-considerations--fairness) for fairness and bias considerations.

## License

[MIT License](./LICENSE)

## Contact & Support

For questions, issues, or feature requests, open an issue in this repository or contact the team.