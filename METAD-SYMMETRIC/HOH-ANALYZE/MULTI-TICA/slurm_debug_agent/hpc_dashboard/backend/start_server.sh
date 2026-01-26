#!/bin/bash
# Start the HPC Dashboard backend server with environment variables

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start the server
exec python -c "import uvicorn; import main; uvicorn.run(main.app, host='0.0.0.0', port=8081)"
