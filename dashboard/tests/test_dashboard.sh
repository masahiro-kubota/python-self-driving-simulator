#!/usr/bin/env bash
# Test dashboard generation end-to-end
#
# This script:
# 1. Builds the React frontend
# 2. Generates test data
# 3. Creates a dashboard HTML file
# 4. Opens it in the browser
#
# Usage:
#   ./dashboard/tests/test_dashboard.sh [--no-build] [--no-open]
#
# Options:
#   --no-build    Skip npm build step (use existing dist/index.html)
#   --no-open     Don't open the browser automatically

set -e  # Exit on error

# Parse arguments
SKIP_BUILD=false
SKIP_OPEN=false

for arg in "$@"; do
    case $arg in
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --no-open)
            SKIP_OPEN=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--no-build] [--no-open]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DASHBOARD_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Dashboard End-to-End Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Build frontend
if [ "$SKIP_BUILD" = false ]; then
    echo "📦 Step 1/3: Building React frontend..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cd "$DASHBOARD_DIR/frontend"
    npm run build
    echo ""
    echo "✓ Frontend built successfully"
    echo ""
else
    echo "⏭️  Step 1/3: Skipping frontend build (--no-build)"
    echo ""
fi

# Step 2: Generate test dashboard
echo "🔧 Step 2/3: Generating test dashboard..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$PROJECT_ROOT"
OUTPUT_FILE="$DASHBOARD_DIR/tests/test_dashboard.html"

uv run python -c "
from pathlib import Path
import sys
sys.path.insert(0, '$DASHBOARD_DIR/tests')

from dummy_data import generate_circular_trajectory, create_experiment_result_from_log
from dashboard.generator import HTMLDashboardGenerator

# Generate test data
print('  → Generating circular trajectory...')
log = generate_circular_trajectory(num_steps=100, radius=50.0)

# Create ExperimentResult from log
print('  → Creating experiment result...')
result = create_experiment_result_from_log(log, experiment_name='Test Dashboard')

# Generate dashboard
output_path = Path('$OUTPUT_FILE')
print(f'  → Creating dashboard HTML...')
generator = HTMLDashboardGenerator()
generator.generate(result, output_path)

print(f'  → Dashboard saved to: {output_path}')
"

echo ""
echo "✓ Dashboard generated successfully"
echo ""

# Step 3: Open in browser
if [ "$SKIP_OPEN" = false ]; then
    echo "🌐 Step 3/3: Opening in browser..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Detect OS and open browser
    if command -v xdg-open &> /dev/null; then
        xdg-open "$OUTPUT_FILE"
        echo "✓ Opened in default browser (Linux)"
    elif command -v open &> /dev/null; then
        open "$OUTPUT_FILE"
        echo "✓ Opened in default browser (macOS)"
    else
        echo "⚠️  Could not detect browser opener"
        echo "   Please open manually: file://$OUTPUT_FILE"
    fi
else
    echo "⏭️  Step 3/3: Skipping browser open (--no-open)"
    echo ""
    echo "📄 Dashboard location:"
    echo "   file://$OUTPUT_FILE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Test completed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
