#!/bin/bash

# Create root directory
mkdir -p enhanced-dcgan-web
cd enhanced-dcgan-web || exit

# Create backend structure
mkdir -p backend/storage/{models,reports,static,training_logs}
touch backend/{main.py,requirements.txt,.env.example,health_check.py,api_client.py}

# Create frontend structure
mkdir -p frontend/public
touch frontend/public/{index.html,favicon.ico}

mkdir -p frontend/src/{components,services,hooks,styles}
touch frontend/src/components/{Dashboard.js,TrainingInterface.js,InteractiveGeneration.js,AnalyticsPanel.js,ReportsPanel.js,LogsPanel.js}
touch frontend/src/services/api.js
touch frontend/src/hooks/{useSystemStatus.js,useTrainingStatus.js,useWebSocket.js}
touch frontend/src/styles/index.css
touch frontend/src/{App.js,index.js}
touch frontend/{package.json,tailwind.config.js,postcss.config.js,.env.development,.env.production}

# Create deployment structure
mkdir -p deployment
touch deployment/{docker-compose.yml,Dockerfile.backend,Dockerfile.frontend,nginx.conf,nginx-frontend.conf,render.yaml}

# Create scripts structure
mkdir -p scripts
touch scripts/{start.sh,deploy.sh,health_check.py}

# Root-level files
touch .env.example .gitignore README.md

echo "✅ Project structure for enhanced-dcgan-web created successfully."
