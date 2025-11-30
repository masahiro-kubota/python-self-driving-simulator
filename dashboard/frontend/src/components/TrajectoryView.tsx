import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSimulationStore } from '../store/simulationStore';
import { useTheme } from '@mui/material';

interface TrajectoryViewProps {
  width: number;
  height: number;
}

export const TrajectoryView: React.FC<TrajectoryViewProps> = ({ width, height }) => {
  const theme = useTheme();
  const { data, getCurrentPoint } = useSimulationStore();
  const currentPoint = getCurrentPoint();

  // Prepare trajectory data
  const trajectoryData = useMemo(() => {
    if (!data || data.steps.length === 0) return [];

    const traces: Plotly.Data[] = [];

    // Map lines (Lanelet)
    if (data.map_lines) {
      data.map_lines.forEach((line, idx) => {
        traces.push({
          x: line.points.map((p) => p.x),
          y: line.points.map((p) => p.y),
          mode: 'lines',
          type: 'scatter',
          name: `Map Line ${idx + 1}`,
          line: {
            color: theme.palette.grey[400],
            width: 1,
          },
          showlegend: false,
          hoverinfo: 'skip',
        });
      });
    }

    // Full trajectory path
    traces.push({
      x: data.steps.map((p) => p.x),
      y: data.steps.map((p) => p.y),
      mode: 'lines',
      type: 'scatter',
      name: 'Trajectory',
      line: {
        color: theme.palette.grey[300],
        width: 2,
      },
      showlegend: false,
      hovertemplate: 'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    });

    // Current position marker
    if (currentPoint) {
      traces.push({
        x: [currentPoint.x],
        y: [currentPoint.y],
        mode: 'markers',
        type: 'scatter',
        name: 'Current Position',
        marker: {
          color: theme.palette.primary.main,
          size: 10,
          symbol: 'circle',
        },
        showlegend: false,
        hovertemplate:
          'X: %{x:.2f}<br>Y: %{y:.2f}<br>Yaw: ' +
          ((currentPoint.yaw * 180) / Math.PI).toFixed(1) +
          '°<extra></extra>',
      });
    }

    return traces;
  }, [data, currentPoint, theme]);

  // Calculate layout with proper aspect ratio
  const layout = useMemo((): Partial<Plotly.Layout> => {
    if (!data || data.steps.length === 0) {
      return {
        width,
        height: height - 40,
        xaxis: { title: { text: 'X (m)' } },
        yaxis: { title: { text: 'Y (m)' }, scaleanchor: 'x', scaleratio: 1 },
        margin: { l: 50, r: 50, t: 20, b: 50 },
        paper_bgcolor: theme.palette.background.paper,
        plot_bgcolor: theme.palette.background.default,
      };
    }

    const xCoords = data.steps.map((p) => p.x);
    const yCoords = data.steps.map((p) => p.y);

    // Include map_lines in bounds calculation
    if (data.map_lines) {
      data.map_lines.forEach((line) => {
        line.points.forEach((point) => {
          xCoords.push(point.x);
          yCoords.push(point.y);
        });
      });
    }

    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    // Add padding
    const padding = Math.max(maxX - minX, maxY - minY) * 0.1;

    return {
      width,
      height: height - 40,
      xaxis: {
        title: { text: 'X (m)' },
        range: [minX - padding, maxX + padding],
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      yaxis: {
        title: { text: 'Y (m)' },
        range: [minY - padding, maxY + padding],
        scaleanchor: 'x',
        scaleratio: 1,
        gridcolor: theme.palette.divider,
        zerolinecolor: theme.palette.divider,
      },
      margin: { l: 50, r: 50, t: 20, b: 50 },
      hovermode: 'closest',
      dragmode: 'pan',
      paper_bgcolor: theme.palette.background.paper,
      plot_bgcolor: theme.palette.background.default,
      font: {
        color: theme.palette.text.primary,
      },
      annotations: currentPoint
        ? [
            {
              x: currentPoint.x,
              y: currentPoint.y,
              ax: currentPoint.x + 20 * Math.cos(currentPoint.yaw),
              ay: currentPoint.y + 20 * Math.sin(currentPoint.yaw),
              xref: 'x',
              yref: 'y',
              axref: 'x',
              ayref: 'y',
              showarrow: true,
              arrowhead: 2,
              arrowsize: 1,
              arrowwidth: 2,
              arrowcolor: theme.palette.error.main,
            },
          ]
        : [],
    };
  }, [data, currentPoint, width, height, theme]);

  const config: Partial<Plotly.Config> = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    scrollZoom: true,
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full bg-white rounded-lg shadow-md">
        No Data
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <Plot data={trajectoryData} layout={layout} config={config} />
    </div>
  );
};
